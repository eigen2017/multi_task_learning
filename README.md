# multi_task_learning

multi task learning model using tensorflow

the raw data is metrics in ecg reports.

the according label represents the diagnoses of doctors.


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


# development log
## shrink the classes
commit 500c0aa46f223ddf033f39b35b152be79006b326  
Date:   Tue May 29 19:58:49 2018 +0800  

base on this commit, i found that label: 3\13\18\21\24 are overfitting, 
which means these classes has high corelationship with the 10 columns data. 
so next step i will shrink the 33 classes to 5 classes,  
and continue to jump into solutions of overfitting problems.  

so after commit[500c0aa46f223ddf033f39b35b152be79006b326],  
let's jump into folder [data_with_shrinked_label],   
and the labels are mapped as:  
3\13\18\21\24   ----->   0/1/2/3/4



commit 8372f6f558538d686d2b117bd276421226252a70  
Author: jakie <jakie@localhost.localdomain>  
Date:   Wed May 30 14:39:54 2018 +0800  







