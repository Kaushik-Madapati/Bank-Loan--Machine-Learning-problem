# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:01:43 2018

@author: nmadapati
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import  classification_report
from sklearn.metrics import  confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


####################################################################
# Preparig  data 
#
##########################################################################
def impute_marriage(gender_marriage):
    gender = gender_marriage[0]
    marrage = gender_marriage[1]
    if pd.isnull(marrage) :
        if gender == "Male":
            return "Yes"
        else:
            return "No"
    else:
        return marrage


def impute_dependents(marraige_dependents):
    marraige = marraige_dependents[0]
    dependent = marraige_dependents[1]
    if pd.isnull(dependent) :
        if marraige == "Yes":
            return '1'
        else:
            return '0'
    else :
        return dependent
    
def convert_dependents(dependents):
    dependent = dependents[0]
    if dependent == '1' :
        return 1
    elif dependent == '2' :
        return 2
    else :
        return 3
    

def impute_LoanAmount(amount_term):
    amount = amount_term[0]
    term = amount_term[1]
    if pd.isnull(amount) :
        if term == 12.0:
            return float(loanAmt_LoanPeriod.iloc[0].values)
        elif term == 36.0:
            return float(loanAmt_LoanPeriod.iloc[1].values)
        elif term == 60.0:
            return float(loanAmt_LoanPeriod.iloc[2].values)
        elif term == 84.0:
            return float(loanAmt_LoanPeriod.iloc[3].values)
        elif term == 120.0:
            return float(loanAmt_LoanPeriod.iloc[4].values)
        elif term == 180.0:
            return float(loanAmt_LoanPeriod.iloc[5].values)
        elif term == 240.0:
            return float(loanAmt_LoanPeriod.iloc[6].values)
        elif term == 300.0:
            return float(loanAmt_LoanPeriod.iloc[7].values)
        elif term == 360.0:
            return float(loanAmt_LoanPeriod.iloc[8].values)
        else:
            return loanAmt_LoanPeriod.iloc[9].values
    else :
        return amount
    
def cleaningData(data) :
    
    ####################################################################
    #Cleaning data 
    #
    ##########################################################################
    data['Gender'] = data['Gender'].apply(lambda x: "Male" if pd.isnull(x) else x)
    data['Married'] = data[['Gender','Married']].apply(impute_marriage, axis=1)
    data['Dependents'] = data[['Married', 'Dependents']].apply(impute_dependents, axis=1)
    #data['Dependents'] = data['Dependents'].apply(convert_dependents)
    data['LoanAmount'] = data[['LoanAmount', 'Loan_Amount_Term']].apply(impute_LoanAmount, axis=1)
    data['Self_Employed'] = data['Self_Employed'].apply(lambda x: "No" if pd.isnull(x) else x)
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].apply(lambda x: 360.0 if pd.isnull(x) else x)
    data['Credit_History'] = data['Credit_History'].apply(lambda x: 1.0 if pd.isnull(x) else x)
    data['LoanAmount'] = data[['LoanAmount', 'Loan_Amount_Term']].apply(impute_LoanAmount, axis=1)
   
    ###########################################################
    #### Converting  categorical variable into dummy variables
    ################################################################
    
    gender = pd.get_dummies(data['Gender'], drop_first=True)
    married = pd.get_dummies(data['Married'], drop_first=True)
    education = pd.get_dummies(data['Education'], drop_first =True)
    propertyArea  = pd.get_dummies(data['Property_Area'], drop_first =True)
    selfEmployed  = pd.get_dummies(data['Self_Employed'], drop_first =True)
    dependents  = pd.get_dummies(data['Dependents'], drop_first =True)
    
    
    data = pd.concat([data,  gender,married,education,propertyArea,selfEmployed,dependents], axis =1)
    
    ###################################################################
    ### Droping categorical  variabla and other other obvious data
    #####################################################################
    
    drop_items = ['Gender','Married', 'Education',
                'Property_Area', 'Loan_ID', 'Self_Employed','Dependents', 'CoapplicantIncome']
               
    
    data.drop(drop_items, axis=1, inplace= True)
    

    
    return data
 

####################################################
## F-regression 
####################################################
    
def logicticRegModel(data) :
    x= input_data
    #y=input_data_raw['Item_Outlet_Sales']
    y= pd.get_dummies(input_data_raw['Loan_Status'], drop_first =True)
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.33, random_state=43)
   
    logicReg = LogisticRegression()
    logicReg.fit(X_train,Y_train)
    print(logicReg)
    predit = logicReg.predict(X_test)
    print(confusion_matrix(Y_test, predit))
    print(classification_report(Y_test, predit))


       
    #print(classification_report(Y_test, predict_sales))
    print('MAE:', metrics.mean_absolute_error(Y_test,predit))
    print('MSE:', metrics.mean_squared_error(Y_test,predit))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test,predit)))
    
    return logicReg.predict(data)


####################################################
## GaussianNB
####################################################
    
def GaussianNBModel(data) :
    x= input_data
    #y=input_data_raw['Item_Outlet_Sales']
    y= pd.get_dummies(input_data_raw['Loan_Status'], drop_first =True)
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.33, random_state=43)
    gnb = GaussianNB()
    
    gnb.fit(X_train,Y_train)
    predit = gnb.predict(X_test)
    print(confusion_matrix(Y_test, predit))
    print(classification_report(Y_test, predit))


       
    #print(classification_report(Y_test, predict_sales))
    print('MAE:', metrics.mean_absolute_error(Y_test,predit))
    print('MSE:', metrics.mean_squared_error(Y_test,predit))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test,predit)))
    
    return gnb.predict(data)



   
####################################################
## F-regression 
####################################################
    
def fRegressionModel(data) :
    x= input_data
    #y=input_data_raw['Item_Outlet_Sales']
    y= pd.get_dummies(input_data_raw['Loan_Status'], drop_first =True)
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.33, random_state=43)
    anova_filter = SelectKBest(f_regression, k=3)
    
    # 2) svm
    clf = svm.SVC()
    anova_lm = make_pipeline(anova_filter, clf)
    anova_lm.fit(X_train, Y_train)
    y_pred = anova_lm.predict(X_test)
    
    print(classification_report(Y_test, y_pred))
    print(confusion_matrix(Y_test, y_pred))

       
    #print(classification_report(Y_test, predict_sales))
    print('MAE:', metrics.mean_absolute_error(Y_test,y_pred))
    print('MSE:', metrics.mean_squared_error(Y_test,y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test,y_pred)))
    
    return anova_lm.predict(data)
   

   
####################################################
## F-regression 
####################################################
    
def RFEModel(data) :
    x= input_data
    #y=input_data_raw['Item_Outlet_Sales']
    y= pd.get_dummies(input_data_raw['Loan_Status'], drop_first =True)
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.33, random_state=43)
    #anova_filter = SelectKBest(f_regression, k=3)
    
    # 2) svm
   # clf = SVR(kernel="linear")
   # anova_lm = make_pipeline(anova_filter, clf)
   
    gnb = LogisticRegression()

    estimator = RFE(gnb,  step=1)
    estimator.fit(X_train, Y_train)
    y_pred = estimator.predict(X_test)
    
    print(classification_report(Y_test, y_pred))
    print(confusion_matrix(Y_test, y_pred))



       
    #print(classification_report(Y_test, predict_sales))
    print('MAE:', metrics.mean_absolute_error(Y_test,y_pred))
    print('MSE:', metrics.mean_squared_error(Y_test,y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test,y_pred)))
    
    return estimator.predict(data)
    
####################################################
## MLP
####################################################
    
def MLPModel(data) :
    x= input_data
    X = StandardScaler().fit_transform(x)
    #y=input_data_raw['Item_Outlet_Sales']
    y= pd.get_dummies(input_data_raw['Loan_Status'], drop_first =True)
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.33, random_state=43)
    #anova_filter = SelectKBest(f_regression, k=3)
    
    # 2) svm
   # clf = SVR(kernel="linear")
   # anova_lm = make_pipeline(anova_filter, clf)
   
    gnb = MLPClassifier(alpha=1)

    gnb.fit(X_train, Y_train)
    y_pred = gnb.predict(X_test)
    
    print(classification_report(Y_test, y_pred))
    print(confusion_matrix(Y_test, y_pred))



       
    #print(classification_report(Y_test, predict_sales))
    print('MAE:', metrics.mean_absolute_error(Y_test,y_pred))
    print('MSE:', metrics.mean_squared_error(Y_test,y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test,y_pred)))
    
    return gnb.predict(data)

# Read Data 
###################################################
input_data_raw = pd.read_csv("train_loan_data.csv")  
input_expected_Data = input_data_raw['Loan_Status']
input_data = input_data_raw.drop('Loan_Status', axis =1 )  
loanAmt_LoanPeriod = input_data[['LoanAmount', 'Loan_Amount_Term']].groupby('Loan_Amount_Term').mean()
input_data = cleaningData(input_data)
#input_data = vif_selection(input_data)

test_data_raw = pd.read_csv("test_loan_data.csv")    
test_data = cleaningData(test_data_raw)
#input_data = vif_selection(input_data)

#####################################################
### Predicting loan status 
###################################################
print("MLPModel \n")
loan_predict = MLPModel(test_data)
print("RFE model \n")
loan_predict = RFEModel(test_data)

######################################################
##  preparing output file 
######################################################
#submission = pd.DataFrame({ "Loan_ID": test_data_raw["Loan_ID"],      
 #                           "Loan_Status": loan_predict })
    
#submission["Loan_Status"] = np.where(submission.Loan_Status ==1,"Y","N")
#submission.to_csv('loan_output_test.csv', index=False)

#df = pd.DataFrame(loan_predict)
#df = df.apply(lambda x: "Y" if x ==1 else "N")


