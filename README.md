# multi_task_learning

## summary

multi task learning model using tensorflow

the raw data is metrics on ecg reports.

the according label represents the diagnoses of doctors.


## data demostration

the label is from 0 to 32. that's 33 classes.


the raw data has 12 columns
first two columns are id.

the following 10 columns are:  
'MDC_ECG_HEART_RATE'  
'MDC_ECG_TIME_PD_PR'  
'MDC_ECG_TIME_PD_QRS'  
'MDC_ECG_TIME_PD_QT'  
'MDC_ECG_TIME_PD_QTc'  
'MDC_ECG_ANGLE_P_FRONT'  
'MDC_ECG_ANGLE_QRS_FRONT'  
'MDC_ECG_ANGLE_T_FRONT'  
'RV5'  
'SV1'  

i shuffled the raw data to:  
shuffled_metrics_in_report_label.csv  
shuffled_metrics_in_report.csv  

then i sliced the shuffled raw data into the data folder  
the data folder includes train\dev\test data


## development procedure log

### about precision and recall  
because this project is based on real medical data,    
so the samples are labeled as imbalanced classes,  
precision and recall estimation is best suit imbalanced sample at this time.  
this paper gave a quick experiment on traditionally ROC and precision-recall,   
and the conclusion shows that,   
precision-recall is more informative when facing imbalanced data.  

### shrink the classes
commit 500c0aa46f223ddf033f39b35b152be79006b326  
Date:   Tue May 29 19:58:49 2018 +0800  

after 10000 epochs, only five classes has precision and recall rate that not tend to 0

base on this commit, i found that label: 3\13\18\21\24 are overfitting, 
which means these classes has high corelationship with the 10 columns data. 
so next step i will shrink the 33 classes to 5 classes,  
and continue to jump into solutions of overfitting problems.  

so after commit[500c0aa46f223ddf033f39b35b152be79006b326],  
let's jump into folder [data_with_shrinked_label],   
and the labels are mapped as:  
3\13\18\21\24   ----->   0/1/2/3/4



### to be overfitting
commit 8372f6f558538d686d2b117bd276421226252a70  
Author: jakie <jakie@localhost.localdomain>  
Date:   Wed May 30 14:39:54 2018 +0800  

after 200 epochs, the five classes overfitted very slowly,  
so i added 3 layers with 100 units per layer,  
and after 6000 epochs, 5 classes all overfitted. 

many articles suggested that,   
initialization of input and do batch norm can accelerate the convergency.    
i applied input init and batch norm at the very first time.    
and the effection has not been estimated,   
because i didn't try the model without initializations of input and without batch norm.   

as to the exploding and vanishing problem,    
i also implemented he-kaiming weights init for relu activation funcs   
i printed the units of the last two layers in this version,    
and seeing that the units value are not too large or too small    

it's worthwhile to mention that, i am using adam optimizer,    
which means it has adventages cause of the momentum and RMS probability mechanism in it.   


### overfitting judgement
it's the very first destination of neuro network adjustment: to be overfitting.  
if u don't know how to judge overfitting, u won't know weather u arrived the first target.  
i judge overfitting easily by precision-recall rate of the last batch of every epoch.  
after thousands of epochs, the output is like:
epoch:[4861]  
precision_rate:  
[ 0.91525424  0.96342856  0.80000001  0.90322578  0.84210527]  
recall_rate:  
[ 0.77142859  0.99059927  0.66666669  0.875       0.94117647]  
f1_score:  
[ 0.83720922  0.97682494  0.72727269  0.88888878  0.88888884]  
epoch:[4862]  
precision_rate:  
[ 0.80327868  0.96351194  0.80000001  0.83870965  0.83765113]  
recall_rate:  
[ 0.89090908  0.98599768  0.5         0.89655173  0.91856062]  
f1_score:  
[ 0.84482753  0.97462511  0.61538458  0.86666656  0.87624204]  


so it's overfitting .  
while the standard way to judge overfitting is give the train set and dev set accurracy a contrast.  
just like:

epoch:[44]
precision_rate:
[ 0.60000002  0.94438136  0.66666669  0.53488374  0.77464789]
[ 0.58031088  0.94480139  0.51428574  0.65151513  0.7652247 ]
recall_rate:
[ 0.94117647  0.97082847  0.66666669  0.95833331  0.88709676]
[ 0.93723851  0.97176331  0.69230771  0.88356167  0.86638135]
f1_score:
[ 0.73282439  0.95742232  0.66666663  0.68656713  0.82706761]
[ 0.71679991  0.95809263  0.59016389  0.74999988  0.81266713]








