# 2022-Data-Classification-LSTM

Predicting Analyte Concentrations from Electrochemical Aptasensor Signals Using LSTM Recurrent Networks

Link: https://doi.org/10.3390/bioengineering9100529

Function "confusion.m" computes the confusion matrix for the classification task and generates the metrics essential for evaluating the classification task. These metrics are accuracy and macro F1 score, as there are 6 distinct different labels.

Script "s1dataAugmentZscore" generates, visualises and saves the weighted augmentation data.

Script "s2ULSTM" is the design of Unidirectional LSTM for the classification of 6 different classes of chemicals, ranging from "No Analyte" to "10 micro Molar".

Script "s3BiLSTM" is the design of Bidirectional LSTM for the classification of 6 different classes of chemicals, ranging from "No Analyte" to "10 micro Molar".

Note that these three different chemicals, 31mer and 35mer Oestradiol and 35mer adenosine, were classified separately in this stage.


