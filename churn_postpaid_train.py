import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
#from xgboost import plot_importance
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from decimal import Decimal
#import pydotplus
import matplotlib.pyplot as plt
from datetime import datetime


import pickle #Save the model
#working_dir =sys.argv[1]
mo_key=sys.argv[1]

working_dir = "/home/admin/mining/churn_postpaid_new"

col_names=[]
col_names=pd.read_csv(working_dir + "/Churn_Postpaid_Control_File.csv")

train_data=[]
model_output=[]

data_files = glob.glob(working_dir + '/training/training_data/*.csv')

tmp=[]
for filename in data_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    tmp.append(df)
if tmp:
    train_data=pd.concat(tmp, axis=0)
    model_output=pd.concat(tmp, axis=0)    
else:
    print('No input data!')
    sys.exit()

X=[]
y=[]

for index,row in col_names.iterrows():
    column_name= row["Column_Name"]
    
    if row['Is_Target']==1:
        y = train_data[column_name]      
        train_data.drop(columns=[column_name],inplace=True)
        
    elif row['isRelevant']==0:
        train_data.drop(columns=[column_name],inplace=True)
     
    elif row['Is_Categorical']==1:
       unique_values = train_data[column_name].unique()
       if unique_values.size==2:
           train_data[column_name] = train_data[column_name].replace(unique_values,[0,1])
           train_data.drop(columns=[column_name],inplace=True)
       else:
           
           values = train_data[column_name]
           dummies = pd.get_dummies(values,prefix=column_name+"_")
           print(dummies.columns)
           dummies.drop(columns = dummies.columns[-1],inplace = True)
           train_data.drop(columns=[column_name],inplace=True)
           train_data = pd.concat([train_data,dummies],axis=1)

print(train_data.head(10))
features = train_data.columns  
print features  
X = train_data

# print(X_scaled)

sc = MinMaxScaler()
X = sc.fit_transform(X)

#X la tap du lieu chi con cac KPI dau vao, y la tap du lieu ket qua
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=49)

print(y_train.value_counts())
print(y_test.value_counts())

# Pipeline
# over = RandomOverSampler(sampling_strategy=0.1)
# under = RandomUnderSampler(sampling_strategy=0.5)
# steps = [('o', over), ('u', under)]
# pipeline = Pipeline(steps=steps)
# X_train, y_train = pipeline.fit_resample(X_train, y_train)

# print('Random Pipeline Sampler')
# (unique, counts) = np.unique(y_train, return_counts=True)
# frequencies = np.asarray((unique, counts)).T
# print(frequencies)
# print(y_test.value_counts())

#----------- Random Forest ------------------
print('Begin RF: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
rf_model = RandomForestClassifier(n_estimators=50,n_jobs=-1,random_state=0)
rf_model.fit(X_train,y_train)

plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 6})
#Feature importances
print rf_model.feature_importances_
sorted_idx = rf_model.feature_importances_.argsort()
print sorted_idx
plt.barh(features[sorted_idx], rf_model.feature_importances_[sorted_idx])
plt.xlabel("Feature Importance")
plt.yticks(fontsize=6)
plt.tight_layout()
plt.savefig(working_dir + '/training/output_data/feature_important_chart_' + mo_key + '.png')

#Feature importances using MDI
importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
forest_importances = pd.Series(importances, index=features)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
ax.figure.savefig(working_dir + '/training/output_data/feature_important_mdi_chart_' + mo_key + '.png')

#
y_pred = rf_model.predict(X_test)

print('======= RF:')
result = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(result)
result1 = classification_report(y_test, y_pred)
print('Classification Report:')
print (result1)
result2 = accuracy_score(y_test, y_pred)
print('Accuracy:',100*result2)
print('')
print('End RF: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print('')
#
pickle.dump(rf_model, open(working_dir + '/training/model_output/Churn_Postpaid_RF_model_T' + mo_key + '.sav','wb'))
model_output["RF_Output"]=rf_model.predict(X)


#----------- End Random Forest --------------

#----------- Decision Tree ------------------

#print('Begin DT: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
#dtree = DecisionTreeClassifier(criterion = "gini",
#            random_state = 100,max_depth=3, min_samples_leaf=5)
#dtree = dtree.fit(X_train, y_train)
#data = tree.export_graphviz(dtree, out_file=None, feature_names=train_data.columns)
#graph = pydotplus.graph_from_dot_data(data)
#graph.write_png(working_dir + '/training/output_data/churn_prepaid_decision_tree.png')
#
#y_pred = dtree.predict(X_test)
#print('======= DT:')
#print("Confusion Matrix: ")
#print(confusion_matrix(y_test, y_pred))
#print("Classification Report:")
#print(classification_report(y_test, y_pred))
#print ("Accuracy : ")
#print(accuracy_score(y_test,y_pred)*100)
#print('')
#print('End RF: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
#print('')
#
#pickle.dump(dtree, open(working_dir + '/training/model_output/Churn_Prepaid_DT_model.sav','wb'))
#
#model_output['DT_Output']=dtree.predict(X)


#-------------End Decision Tree --------------------

# ************************** XGBoost *********************************
#print('Begin XGBoost: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
#xgb_clf = XGBClassifier(objective = 'binary:logistic',
#                    max_depth = 4,
#                    alpha = 10,
#                    learning_rate= 1.0,
#                    n_estimators=100)
#xgb_clf.fit(X_train, y_train) # training the model
#
#sorted_idx = xgb_clf.feature_importances_.argsort()
#plt.barh(features, xgb_clf.feature_importances_[sorted_idx])
#plt.xlabel("Xgboost Feature Importance")
#plt.yticks(fontsize=12)
#plt.tight_layout()
#plt.savefig(working_dir + '/training/output_data/xgboost_feature_important_chart.png')
#plt.clf()
#
#y_pred = xgb_clf.predict(X_test)
#
#print('======= XGBoost:')
#result = confusion_matrix(y_test, y_pred)
#print('Confusion Matrix:')
#print(result)
#result1 = classification_report(y_test, y_pred)
#print('Classification Report:',)
#print (result1)
#result2 = accuracy_score(y_test, y_pred)
#print('Accuracy:',100*result2)
#print('')
#print('End XGBoost: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
#print('')
#
#pickle.dump(dtree, open(working_dir + '/training/model_output/Simbox_XGB_model.sav','wb'))
#
#model_output['XBG_Output']=dtree.predict(X)

# ************************** End XGBoost *******************************

# # ************************** Logistic Regression *********************************
# print('Begin GLM: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
# glm_model = LogisticRegression(max_iter = 1000,solver='newton-cg',n_jobs=-1)
# glm_model.fit(X_train,y_train) # training the model

# y_pred = glm_model.predict(X_test)

# print('======= GLM:')
# result = confusion_matrix(y_test, y_pred)
# print('Confusion Matrix:')
# print(result)
# result1 = classification_report(y_test, y_pred)
# print('Classification Report:',)
# print (result1)
# result2 = accuracy_score(y_test, y_pred)
# print('Accuracy:',100*result2)
# print('')
# print('End GLM: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
# print('')

# pickle.dump(dtree, open(working_dir + '/training/model_output/Simbox_GLM_model.sav','wb'))

# model_output['GLM_Output']=dtree.predict(X)


# # ************************** End Logistic Regression *******************************

# # ************************** SVM ********************************
# print('Begin SVM: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
# svm = SVC(kernel= 'linear', random_state=1, C=0.1, probability=True)
# svm.fit(X_train, y_train)
 
# y_pred = svm.predict(X_test)

# print("Results SVM")
# print("Confusion Matrix: ")
# print(confusion_matrix(y_test, y_pred))
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# print ("Accuracy : ")
# print(accuracy_score(y_test,y_pred)*100)
# print('')
# print('End GLM: ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
# print('')

# pickle.dump(svm, open(working_dir + '/training/model_output/Simbox_SVM_model.sav','wb'))
# model_output['SVM_Output']=svm.predict(X)

# # ************************** End SVM ********************************

#----Save output to CSV
#model_output.to_csv(working_dir + '/training/output_data/churn_prepaid_train_output_data.csv')

#with open(working_dir + '/training/.success', 'w') as fp:
#    pass