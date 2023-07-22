import sys
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from sklearn.preprocessing import MinMaxScaler
import os
import pickle #Save the model

#working_dir =sys.argv[1]
mo_key=sys.argv[1]
working_dir = "/home/admin/mining/churn_postpaid_new"

col_names=[]
col_names=pd.read_csv(working_dir + '/Churn_Postpaid_Control_File.csv')

rf_model_filename='Churn_Postpaid_RF_model.sav'
dt_model_filename='Churn_Postpaid_DT_model.sav'
xgb_model_filename='Churn_Postpaid_XGB_model.sav'
glm_model_filename='Churn_Postpaid_model.sav'
svm_model_filename='Churn_Postpaid_model.sav'

predict_data=[]
predict_output=[]
data_files = glob.glob(working_dir+"/predict/predict_data/*.csv")


tmp=[]
for filename in data_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    tmp.append(df)
if tmp:
    predict_data=pd.concat(tmp, axis=0)
    predict_output=pd.concat(tmp, axis=0)
else:
    print('No predict data!')
    sys.exit()

print('Loaded predict data: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

for index,row in col_names.iterrows():
    column_name= row["Column_Name"]
    
    if row['Is_Target']==1:
        y = predict_data[column_name]      
        predict_data.drop(columns=[column_name],inplace=True)
        
    elif row['isRelevant']==0:
        predict_data.drop(columns=[column_name],inplace=True)
     
    elif row['Is_Categorical']==1:
       unique_values = predict_data[column_name].unique()
       if unique_values.size==2:
           predict_data[column_name] = predict_data[column_name].replace(unique_values,[0,1])
           predict_data.drop(columns=[column_name],inplace=True)
       else:
           
           values = predict_data[column_name]
           dummies = pd.get_dummies(values,prefix=column_name+"_")
           print(dummies.columns)
           dummies.drop(columns = dummies.columns[-1],inplace = True)
           predict_data.drop(columns=[column_name],inplace=True)
print(predict_data.head(10))

sc = MinMaxScaler()
scale_data = sc.fit_transform(predict_data)

# ----------------Predicting for the entire training and testing dataset------------

#--------Apply RF model
#model_path = working_dir + '/training/model_output/' + rf_model_filename
model_path = working_dir + '/training/model_output/Churn_Postpaid_RF_model_T202210.sav'
loaded_model = pickle.load(open(model_path, 'rb'))

y_pred_TN = loaded_model.predict(scale_data)
y_pred_proba=loaded_model.predict_proba(scale_data)

predict_output["CHURN_INDEX"]=y_pred_TN

#predict_output["RF_Probability"]=np.max(y_pred_proba) * 100
#print(y_pred_proba.max(axis=1))
probas = y_pred_proba.max(axis=1)

arr_proba = []
arr_stsf_level = []


#for x in probas:  
#  arr_proba.append(x*100)
#  if (x*100) >= 90:  
#    arr_stsf_level.append(1)     
#  elif (x*100) >= 80:
#    arr_stsf_level.append(2)
#  elif (x*100) >= 70:
#    arr_stsf_level.append(3)
#  elif (x*100) >= 60:
#    arr_stsf_level.append(4)      
#  else: 
#    arr_stsf_level.append(5)
#print('result')  

for idx, x in enumerate(probas):
    #print(x)
    output = y_pred_TN[idx]
    value = x*100
    arr_proba.append(value)
    if output == 1:
      if (value) >= 90:  
        arr_stsf_level.append(1)     
      elif (value) >= 80:
        arr_stsf_level.append(2)
      elif (value) >= 70:
        arr_stsf_level.append(3)
      elif (value) >= 60:
        arr_stsf_level.append(4)      
      else: 
        arr_stsf_level.append(5)
    else:
      if (value) >= 90:  
        arr_stsf_level.append(5)     
      elif (value) >= 80:
        arr_stsf_level.append(4)
      elif (value) >= 70:
        arr_stsf_level.append(3)
      elif (value) >= 60:
        arr_stsf_level.append(2)      
      else: 
        arr_stsf_level.append(1)
    #print(output)
    #print(idx, x)
      
predict_output["CHURN_RATE"]=arr_proba
predict_output["SATISFIED_LEVEL"]=arr_stsf_level
predict_output["MO_KEY"]=[mo_key] * len(arr_stsf_level)  

  
#------------

#------Apply DT model
#model_path = working_dir + '/predict/model/' + dt_model_filename
#loaded_model = pickle.load(open(model_path, 'rb'))


#y_pred_TN = loaded_model.predict(scale_data)
#y_pred_proba=loaded_model.predict_proba(scale_data)

#predict_output["DT_Output"]=y_pred_TN
#predict_output["DT_Probability"]=np.max(y_pred_proba) * 100
#-------------

#------Apply XGB model
#model_path = working_dir + '/predict/model/' + xgb_model_filename
#loaded_model = pickle.load(open(model_path, 'rb'))


#y_pred_TN = loaded_model.predict(scale_data)
#y_pred_proba=loaded_model.predict_proba(scale_data)

#predict_output["XGB_Output"]=y_pred_TN
#predict_output["XGB_Probability"]=np.max(y_pred_proba) * 100
#-------------

# #-----Apply GLM model
# model_path = working_dir + '/predict/model/' + glm_model_filename
# loaded_model = pickle.load(open(model_path, 'rb'))


# y_pred_TN = loaded_model.predict(scale_data)
# y_pred_proba=loaded_model.predict_proba(scale_data)

# predict_output["GLM_Output"]=y_pred_TN
# predict_output["GLM_Probability"]=np.max(y_pred_proba) * 100
# #-------------

# #Apply RF model
# model_path = working_dir + '/predict/model/' + svm_model_filename
# loaded_model = pickle.load(open(model_path, 'rb'))


# y_pred_TN = loaded_model.predict(scale_data)
# y_pred_proba=loaded_model.predict_proba(scale_data)

# predict_output["SVM_Output"]=y_pred_TN
# predict_output["SVM_Probability"]=np.max(y_pred_proba) * 100

#predict_output.drop(columns=["day_key"],inplace=True)


path = working_dir + '/predict/predict_output/' + mo_key

# Check whether the specified path exists or not
isExist = os.path.exists(path)

if not isExist:
  
  # Create a new directory because it does not exist 
  os.makedirs(path)
  print("The new directory is created!")
  
  
predict_output.drop(predict_output.columns.difference(['SERVICE_NBR','ACCT_SRVC_INSTANCE_KEY','ACCT_KEY','CUST_KEY','PROD_LINE_KEY','BSNS_RGN_KEY','GEO_STATE_CD','GEO_DSTRCT_CD','GEO_CITY_CD','CHURN_INDEX','CHURN_RATE','SATISFIED_LEVEL','MO_KEY']), axis=1, inplace=True)
predict_output.to_csv(working_dir + '/predict/predict_output/' + mo_key + '/churn_postpaid_predict.csv', index=False)
df = pd.read_csv(working_dir + '/predict/predict_output/' + mo_key + '/churn_postpaid_predict.csv')
#predict_output.to_csv(working_dir + '/predict/predict_output/churn_postpaid_predict_T'+ mo_key + '_T11.csv', index=False)
#df = pd.read_csv(working_dir + '/predict/predict_output/churn_postpaid_predict_T'+ mo_key + '_T11.csv')
# table = pa.Table.from_pandas(predict_output, schema=predicted_output_schema, preserve_index=False)
# pq.write_to_dataset(table, working_dir + "/tmp/output_data")
# df = pd.read_parquet(working_dir + "/tmp/output_data", engine='pyarrow')

print(df.head(5))
print('Model path:' + model_path)
print('Total record: ' + str(len(df)))
print('ChurnInd 1: ' + str(len(df[df['CHURN_INDEX']==1])))
print('ChurnInd 0: ' + str(len(df[df['CHURN_INDEX']==0])))
total=len(df)
a=len(df[df['CHURN_INDEX']==1])
b=len(df[df['CHURN_INDEX']==0])
print(total)
print(a)
print(b)
print("%.2f" % ((a/total)*100))
print('Rate: ' + str((a/total)*100))

with open(working_dir + '/predict/.success', 'w') as fp:
    pass

