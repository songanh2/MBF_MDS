#!/bin/bash

if [ "$1"A = A ]
then
    echo 'First argument mo_key is null or empty.'
    exit 0
else
    mo_key=$1
fi

#Save  logs
LOG_LOCATION=/home/admin/mining/churn_postpaid_new/logs/
LOG_FILE_NAME=${LOG_LOCATION}/churn_postpaid_predict_${mo_key}_T111111.log
exec > >(tee -i $LOG_FILE_NAME)
exec 2>&1

working_dir='/home/admin/mining/churn_postpaid_new'
hdfs_dir='/DATALAKE_TLS_TEST/EXP_CHURN_POSTPAID_PREDICT_DATA/'$mo_key
log_dir='/home/admin/mining/churn_postpaid_new/logs'

#Kinit Keberos
printf 'Mbf@1234' | kinit

#Export data to HDFS
cd $working_dir
spark-submit --driver-memory 20g --num-executors 10 --executor-cores 5 --executor-memory 5g  exp_churn_postpaid_predict_data_T07.py $mo_key

#Get data from HDFS to Local
rm $working_dir/predict/predict_data/*
hdfs dfs -copyToLocal -f $hdfs_dir/*.csv $working_dir/predict/predict_data/

#Load model to apply data
#python churn_postpaid_predict.py > /backup/mining/churn_prepaid/logs/churn_postpaid_train.log
python churn_postpaid_predict.py $mo_key

echo '------- '$(date '+%Y-%m-%d %H:%M:%S')' ----------'

