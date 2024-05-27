%%% Data Augmentation 1 With Normalized Signals For Loop - z score
%%% CNTFET_90s_Alsager_35mer_Oestradiol




% rng(123);
clc; close all; clear;

% n is the number of signals that is going to be generated
N = 100;


%% Import Segments
structPath = "Data\SegmentStruct.mat";
load(structPath)

% Permutation for geneating 100 segments - Manually choosing concentrations
% The analyte concentration is chosen manually here

for k = 1:size(s,2)

    T = s(k).zscore;
    A = table2array(T);
    concenTitle = s(k).AnalyteConcentration;
    
    tableVars = size(T,2);
    n = N- tableVars;
    v = 1:tableVars;
    combinationList = nchoosek(v,2);
    combinationSize = size(combinationList, 1);
    
    % random number list generated for selecting the combinationList
    rc = randi(combinationSize,1,n);
     
    % 100 pairs of number
    
    rcList = combinationList(rc,:);
    rcList1 = rcList(:,1);
    rcList2 = rcList(:,2);
    
    segment1= A(:,rcList1)';
    segment2= A(:,rcList2)';
    
    segment1Name = string(T.Properties.VariableNames(rcList1));
    segment2Name = string(T.Properties.VariableNames(rcList2));
    
    checkDifferentSegment(segment1,segment2, segment1Name, segment2Name)
    
    % random weight vectors
    rw = randomWeight(n);
    rw1 = rw(:,1);
    rw2 = rw(:,2);
    
    
    randomWeightedSum = rw1 .* segment1 + rw2 .* segment2;
    VarNameSegments = strcat(string(rw1),'*',segment1Name' ,'+',string(rw2),'*',segment2Name');

    randomWeightSumTable = array2table(randomWeightedSum');
    randomWeightSumTable.Properties.VariableNames = VarNameSegments;

    AllSegments = [T randomWeightSumTable];
    
    %% Plot some generated segments
    
    
    plotNumber = 2;
    rp = randperm(n,plotNumber);
    
    for i=1: plotNumber
    
        m= rp(i); % the selected segment for Plot
    
        
        s1= segment1(m,:);
        s2= segment2(m,:);
        w1= rw1(m);
        w2= rw2(m);
        s1Name = strcat('S1:',segment1Name(m));
        s2Name = strcat('S2:',segment2Name(m));
        sfinal= randomWeightedSum(m,:);
        sfinalName = strcat('Augmented Segment- ',string(m));
    
        
        fig= figure;
        fig.Position = [10 10 1200 600];
        figNameText= strcat('Normalized(zscore) Signals - ',concenTitle, '- Augmented segment-',string(m));
        fig.Name = figNameText;
    
        
        
        plot(s1,'r','DisplayName',s1Name,'LineWidth',1.2);
        hold on
        plot(s2,'b','DisplayName',s2Name,'LineWidth',1.2);
        plot(sfinal,'g','DisplayName',sfinalName,'LineWidth',1.2);
        hold off
    
        legend("Location","eastoutside","Interpreter","latex");
    
        titleText = strcat('Normalized(zscore) Signals - ',concenTitle, ' - w1 =  ',string(w1),' - w2 =  ',string(w2));
        title(titleText,"Interpreter","latex");
    
        xlabel("Time","Interpreter","latex")
        ylabel("Normalized Drain Current (zscore)","Interpreter","latex");
    end
end


% Save Plots
fig = findobj('Type', 'figure');
for i=1:length(fig)
    name = strcat(fig(i).Name,'.png');        
    figPath = "Image\CNTFET_90s_Alsager_35mer_Oestradiol-AugmentedSignals";
    saveas(fig(i),fullfile(figPath,name));       
end

% Save Structure 
filepath = "Data\";
structureName = strcat(filepath,'\','SegmentStruct','.mat');

save(structureName,'s');


% Function: Check segments to be different
function  checkDifferentSegment(Segment1,Segment2, Segment1Name, Segment2Name)
    flag1 = Segment1 == Segment2;
    flag2 = Segment1Name == Segment2Name;
    
    flag1 = nnz(flag1);
    flag2 = nnz(flag2);
    
    if flag1>0 || flag2>0
        disp("An identical segment is chosen for random weight sum ");
    
    else
        disp("Two segments chosen for random weighted sum are different");
    end
end

% Function: Generate Random Weights
function w = randomWeight(n)
    w= abs(randn(n,2));
    w = bsxfun(@rdivide, w,sum(w,2));
end

