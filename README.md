# multi_task_learning

multi task learning model using tensorflow

the raw data is metrics in ecg reports.

the according label represents the diagnoses of doctors.


the label is from 0 to 32. that's 33 classes.

the first 2 columns of label file and data file is sample id, 

the sample in label file and data file are strictly aligned, 

so u can just ignore the first 2 columns.



the raw data has 12 columns
first two columns are id.

the following 10 columns are:&nbsp;
'MDC_ECG_HEART_RATE'&nbsp;
'MDC_ECG_TIME_PD_PR'&nbsp;
'MDC_ECG_TIME_PD_QRS'&nbsp;
'MDC_ECG_TIME_PD_QT'&nbsp;
'MDC_ECG_TIME_PD_QTc'&nbsp;
'MDC_ECG_ANGLE_P_FRONT'&nbsp;
'MDC_ECG_ANGLE_QRS_FRONT'&nbsp;
'MDC_ECG_ANGLE_T_FRONT'&nbsp;
'RV5'&nbsp;
'SV1'&nbsp;

i shuffled the raw data to shuffled_files
