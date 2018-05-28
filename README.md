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
