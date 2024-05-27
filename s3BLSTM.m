%LSTM for Oestradiol data- 31mer
%One Segment One Sequence- Test result




clc;clear;close all;
rng(123)
tic

structPathTrain = "Data\35merAdenosineSegmentStructTrain.mat";
load(structPathTrain)



structPathTest = "Data\35merAdenosineSegmentStructTest.mat";
load(structPathTest)


tablePath = "Result\LSTM-35merAdenosine\";

%Making input and response from original data sets
%numFeatures= 1 means just one part of segment (e.g. inputSequenceLength = 30, means just 30 sec of specific segment)
%numFeatures= 0 means just all data points of a segment as the input (e.g. inputSequenceLength = 30, means all data 
 %of a segment will divided to 30 sec parts

%Train Structure
%make the data for net
numInputFeature = 500;
numFeatures = 0; 

inputSequenceCell=[];
response=[];

for j=1:size(sTrain,2)
    T = sTrain(j).zscore100;  
    loopResponseLabel = j;

    [loopInputSequenceCell, loopResponse] = makeInputSeqResponseLabel(...
    T, numInputFeature, loopResponseLabel, numFeatures);

    inputSequenceCell = [inputSequenceCell; loopInputSequenceCell];    
    response=[response; loopResponse];
end

data = inputSequenceCell;

%Test Structure
%make the data checking the accuracy from the untouched data
numInputFeature = 500;
numFeatures = 0; 

inputSequenceCellTest=[];
responseTest=[];

for j=1:size(sTest,2)
    T = sTest(j).zscore;  
    loopResponseLabel = j;

    [loopInputSequenceCell, loopResponse] = makeInputSeqResponseLabel(...
    T, numInputFeature, loopResponseLabel, numFeatures);

    inputSequenceCellTest = [inputSequenceCellTest; loopInputSequenceCell];    
    responseTest=[responseTest; loopResponse];
end

dataTest = inputSequenceCellTest;

%Splitting Data to Train, Validation, Test sets
rng(123)

trainRatio = 0.6;
valRatio = 0.2;
testRatio = 0.2;

[XTrain, XValid, XTest, YTrain, YValid, YTest]  = dataSplit(data, response, trainRatio,valRatio,testRatio);
YTestDouble = double(YTest);
YTestVector = full(ind2vec(YTestDouble',6));


XHatTest = dataTest;
YHatTest = responseTest;
YHatTestDouble= double(YHatTest);
YHatTestVector = full(ind2vec(YHatTestDouble',6));

maxEpochs = 50;

miniBatchSize = 20;
validFreq = 20;
initialLearnRate = 0.005;
learnRateDropFac = 0.2;
learnRateDropPer = 5;
l2Reg= 0.01;
gradientDecayFac = 0.9;
squaredGradientDecayFac = 0.9;
plotsStr = 'training-progress';


numInputFeature = size(XTrain{1},1);
numHiddenUnitsAll = [50; 100; 150 ; 200 ; 250 ;300 ;350 ; 400 ; 450 ; 500];
% numHiddenUnitsAll= 150;

labelCountTrain= countlabels(YTrain);
labelCountValid = countlabels(YValid);
labelcountTest = countlabels(YTest);

%LSTM for Classification
numClasses =6;

varNameHidden = string(numHiddenUnitsAll)';

accuracyTableReal = table();
microF1TableReal = table();
macroF1TableReal = table();

accuracyTableNet = table();
microF1TableNet = table();
macroF1TableNet = table();


VarName = ["Options" varNameHidden];


set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

accuracyTableReal(1,1) = {"ADAM"};
microF1TableReal(1,1) = {"ADAM"};
macroF1TableReal(1,1) = {"ADAM"};

accuracyTableNet(1,1) = {"ADAM"};
microF1TableNet(1,1) = {"ADAM"};
macroF1TableNet(1,1) = {"ADAM"};


optimiser='ADAM';

for j=1: size(numHiddenUnitsAll)
    numHiddenUnits = numHiddenUnitsAll(j);

    layers = [ ...
    sequenceInputLayer(numInputFeature)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];   

    options1 = trainingOptions('adam',...      
        'ResetInputNormalization',false,...   
        'InitialLearnRate',initialLearnRate,...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropFactor',learnRateDropFac,...
        'LearnRateDropPeriod',learnRateDropPer,...
        'GradientDecayFactor',gradientDecayFac,...
        'SquaredGradientDecayFactor',squaredGradientDecayFac,...
        'L2Regularization',l2Reg,...        
        'MaxEpochs',maxEpochs, ...,
        'MiniBatchSize',miniBatchSize,...
        'ValidationData',{XValid,YValid}, ...
        'ValidationFrequency',validFreq, ...         
        'Shuffle','every-epoch', ...
        'Verbose',0, ...
        'Plots',plotsStr,...
        'OutputNetwork','best-validation-loss');



    [net1,info1] = trainNetwork(XTrain ,YTrain,layers,options1);

    
    YPred1 = classify(net1,XHatTest);
    YPred1net = classify(net1,XTest);    


    YPred1Double = double(YPred1);
    YPred1Vector = full(ind2vec(YPred1Double',numClasses));

    YPred1netDouble = double(YPred1net);
    YPred1netVector= full(ind2vec(YPred1netDouble',numClasses));


    [c_matrix, Result, RefereceResult] = confusion.getMatrix(YHatTestDouble,YPred1Double,0);
    [c_matrix2, Result2, RefereceResult2] = confusion.getMatrix(YTestDouble,YPred1netDouble,0);


    totalTP = sum(RefereceResult.TruePositive);
    totalFP = sum(RefereceResult.FalsePositive);
    totalFN = sum(RefereceResult.FalseNegative);

    totalTP2 = sum(RefereceResult2.TruePositive);
    totalFP2 = sum(RefereceResult2.FalsePositive);
    totalFN2 = sum(RefereceResult2.FalseNegative);




    accuracyTableReal(1,j+1) = num2cell(Result.Accuracy);
    macroF1TableReal(1,j+1) = num2cell(Result.F1_score);
    microF1TableReal(1,j+1) = num2cell(totalTP/(totalTP+(0.5*(totalFP+totalFN))));

    accuracyTableNet(1,j+1) = num2cell(Result2.Accuracy);
    macroF1TableNet(1,j+1) = num2cell(Result2.F1_score);
    microF1TableNet(1,j+1) = num2cell(totalTP2/(totalTP2+(0.5*(totalFP2+totalFN2))));
  
end

accuracyTableReal.Properties.VariableNames = VarName;
microF1TableReal.Properties.VariableNames = VarName;
macroF1TableReal.Properties.VariableNames = VarName;

accuracyTableNet.Properties.VariableNames = VarName;
microF1TableNet.Properties.VariableNames = VarName;
macroF1TableNet.Properties.VariableNames = VarName;


accuracyTableReal(2,1) = {"RMSPROP"};
microF1TableReal(2,1) = {"RMSPROP"};
macroF1TableReal(2,1) = {"RMSPROP"};



accuracyTableNet(2,1) ={"RMSPROP"};
microF1TableNet(2,1) = {"RMSPROP"};
macroF1TableNet(2,1) = {"RMSPROP"};


optimiser = 'RMSPROP';

for j=1: size(numHiddenUnitsAll)
    numHiddenUnits = numHiddenUnitsAll(j);

    layers = [ ...
    sequenceInputLayer(numInputFeature)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

    

    options2 = trainingOptions('rmsprop', ...    
    'ResetInputNormalization',false,...   
    'InitialLearnRate',initialLearnRate,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',learnRateDropFac,...
    'LearnRateDropPeriod',learnRateDropPer,...
    'SquaredGradientDecayFactor',0.9,...
    'L2Regularization',l2Reg,...
    'MaxEpochs',maxEpochs, ...,
    'MiniBatchSize',miniBatchSize,...
    'ValidationData',{XValid,YValid}, ...
    'ValidationFrequency',validFreq, ...
    'Shuffle','every-epoch', ...
    'Verbose',0, ...
    'Plots',plotsStr,...
    'OutputNetwork','best-validation-loss');


    [net2 ,info2]= trainNetwork(XTrain ,YTrain,layers,options2); 

    YPred2 = classify(net2,XHatTest);
    YPred2net = classify(net2,XTest);

    


    YPred2Double = double(YPred2);
    YPred2Vector = full(ind2vec(YPred2Double',numClasses));

    YPred2netDouble = double(YPred2net);
    YPred2netVector= full(ind2vec(YPred2netDouble',numClasses));


    [c_matrix, Result, RefereceResult] = confusion.getMatrix(YHatTestDouble,YPred2Double,0);
    [c_matrix2, Result2, RefereceResult2] = confusion.getMatrix(YTestDouble,YPred2netDouble,0);


    totalTP = sum(RefereceResult.TruePositive);
    totalFP = sum(RefereceResult.FalsePositive);
    totalFN = sum(RefereceResult.FalseNegative);

    totalTP2 = sum(RefereceResult2.TruePositive);
    totalFP2 = sum(RefereceResult2.FalsePositive);
    totalFN2 = sum(RefereceResult2.FalseNegative);




    accuracyTableReal(2,j+1) = num2cell(Result.Accuracy);
    macroF1TableReal(2,j+1) = num2cell(Result.F1_score);
    microF1TableReal(2,j+1) = num2cell(totalTP/(totalTP+(0.5*(totalFP+totalFN))));

    accuracyTableNet(2,j+1) = num2cell(Result2.Accuracy);
    macroF1TableNet(2,j+1) = num2cell(Result2.F1_score);
    microF1TableNet(2,j+1) = num2cell(totalTP2/(totalTP2+(0.5*(totalFP2+totalFN2))));

end

accuracyTableReal(3,1) = {"SGDM"};
microF1TableReal(3,1) = {"SGDM"};
macroF1TableReal(3,1) = {"SGDM"};

accuracyTableNet(3,1) ={"SGDM"};
microF1TableNet(3,1) = {"SGDM"};
macroF1TableNet(3,1) = {"SGDM"};

optimiser = 'SGDM';


for j=1: size(numHiddenUnitsAll)
    numHiddenUnits = numHiddenUnitsAll(j);

    layers = [ ...
    sequenceInputLayer(numInputFeature)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
    

    options3 = trainingOptions('sgdm', ...    
    'ResetInputNormalization',false,...   
    'InitialLearnRate',initialLearnRate,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',learnRateDropFac,...
    'LearnRateDropPeriod',learnRateDropPer,...        
    'L2Regularization',l2Reg,...    
    'MaxEpochs',maxEpochs, ...,
    'MiniBatchSize',miniBatchSize,...
    'ValidationData',{XValid,YValid}, ...
    'ValidationFrequency',validFreq, ...
    'Shuffle','every-epoch', ...
    'Verbose',0, ...
    'Plots',plotsStr,...
    'OutputNetwork','best-validation-loss');


    [net3,info3] = trainNetwork(XTrain ,YTrain,layers,options3);

    
    YPred3 = classify(net3,XHatTest);
    YPred3net = classify(net3,XTest);   

    YPred3Double = double(YPred3);
    YPred3Vector = full(ind2vec(YPred3Double',numClasses));

    YPred3netDouble = double(YPred3net);
    YPred3netVector= full(ind2vec(YPred3netDouble',numClasses));

    [c_matrix, Result, RefereceResult] = confusion.getMatrix(YHatTestDouble,YPred3Double,0);
    [c_matrix2, Result2, RefereceResult2] = confusion.getMatrix(YTestDouble,YPred3netDouble,0);


    totalTP = sum(RefereceResult.TruePositive);
    totalFP = sum(RefereceResult.FalsePositive);
    totalFN = sum(RefereceResult.FalseNegative);

    totalTP2 = sum(RefereceResult2.TruePositive);
    totalFP2 = sum(RefereceResult2.FalsePositive);
    totalFN2 = sum(RefereceResult2.FalseNegative);

    accuracyTableReal(3,j+1) = num2cell(Result.Accuracy);
    macroF1TableReal(3,j+1) = num2cell(Result.F1_score);
    microF1TableReal(3,j+1) = num2cell(totalTP/(totalTP+(0.5*(totalFP+totalFN))));

    accuracyTableNet(3,j+1) = num2cell(Result2.Accuracy);
    macroF1TableNet(3,j+1) = num2cell(Result2.F1_score);
    microF1TableNet(3,j+1) = num2cell(totalTP2/(totalTP2+(0.5*(totalFP2+totalFN2))));

    
end

%Save Tables
accuracyTableName= strcat(tablePath, 'AccuracyReal-zscoreAugment-1seg1seqBiLSTM3NetsHiddenUnEpoch50.csv');
writetable(accuracyTableReal, accuracyTableName);

% microF1TableName= strcat(tablePath, 'microF1Real-zscoreAugment-1seg1seqBiLSTM3NetsHiddenUnEpoch50.csv');
% writetable(microF1TableReal, microF1TableName);

macroF1TableName= strcat(tablePath, 'macroF1Real-zscoreAugment-1seg1seqBiLSTM3NetsHiddenUnEpoch50.csv');
writetable(macroF1TableReal, macroF1TableName);

accuracyTableName= strcat(tablePath, 'AccuracyNet-zscoreAugment-1seg1seqBiLSTM3NetsHiddenUnEpoch50.csv');
writetable(accuracyTableNet, accuracyTableName);

% microF1TableName= strcat(tablePath, 'microF1Net-zscoreAugment-1seg1seqBiLSTM3NetsHiddenUnEpoch50.csv');
% writetable(microF1TableNet, microF1TableName);

macroF1TableName= strcat(tablePath, 'macroF1Net-zscoreAugment-1seg1seqBiLSTM3NetsHiddenUnEpoch50.csv');
writetable(macroF1TableNet, macroF1TableName);



%Function Making Input Sequence and Response Label: makeInputSeqResponseLabel
function [inputSequenceCell,response] = makeInputSeqResponseLabel(...
    T, inputSequenceLength, responseLabel, numFeatures)

   
    inputSequenceCell ={};  
    inputMatAll =[];
    % ***************
    if numFeatures == 1    
        for i = 1:size(T,2)
            loopVec = T.(i);
            [loopMat,~] = matrixMaker(loopVec,inputSequenceLength);
            inputMatAll =[inputMatAll ;loopMat];
            loopResponse = repmat(responseLabel,size(inputMatAll,1),1);
            loopResponseCategory = categorical(loopResponse);    
        end
        
        inputSequenceCell = num2cell(inputMatAll,2);
        response= repmat(responseLabel,size(inputMatAll,1),1);
        response = categorical(response);  
    % ***************

    % ***************
    else        

        for i = 1:size(T,2)
            loopVec = T.(i);
            [loopMatTableVar,~] = matrixMaker(loopVec,inputSequenceLength);
            inputMatAll = [inputMatAll ; {loopMatTableVar}];
        end
        inputSequenceCell = inputMatAll;
        response= repmat(responseLabel,size(inputMatAll,1),1);
        response = categorical(response);     
    % ***************
    end     % if-else
end     % function   
%Function Train, Valid, Test blocks maker: dataSplit
function [trainData, validData, testData, responseTrain, responseValid, responseTest]  = dataSplit(data, response, trainRatio,valRatio,testRatio)

    [~,label,~] = groupcounts(response);
    
    trainData =[];
    responseTrain= [];
    
    validData = [];
    responseValid =[];
    
    testData = [];
    responseTest=[];
    
    for i = 1: size(label, 1)
    
        idxLoop = find(double(response) == i);
        responseLoop = response(idxLoop);
    
        dataLoop = data(idxLoop);
        dataLoopSize = size(dataLoop,1);
    
        [trainIndLoop,valIndLoop,testIndLoop] = dividerand(dataLoopSize,trainRatio,valRatio,testRatio);
    
        trainLoop = dataLoop(trainIndLoop);
        responseTrainLoop = responseLoop(trainIndLoop);
    
        validLoop = dataLoop(valIndLoop);
        responseValidLoop = responseLoop(valIndLoop);
    
        testLoop = dataLoop(testIndLoop);
        responseTestLoop = responseLoop(testIndLoop);
    
        trainData=[trainData; trainLoop];
        responseTrain = [responseTrain; responseTrainLoop];
    
        validData = [validData; validLoop];
        responseValid = [responseValid;responseValidLoop];
    
        testData = [testData; testLoop];
        responseTest = [responseTest; responseTestLoop];   
    end
end
%Function Reshape the Segmant Vectors/ matrixMaker
function [outputMatrix,outputMatrixCell] = matrixMaker(vector, inputSequenceLength)

    numColumn= floor(length(vector)/inputSequenceLength);
    endPoint = numColumn * inputSequenceLength;
    vector = vector(1:endPoint);      
    outputMatrixTranspose = reshape(vector,[inputSequenceLength,numColumn]);
    outputMatrix = outputMatrixTranspose';
    outputMatrixCell = num2cell(outputMatrix,2);    

end

