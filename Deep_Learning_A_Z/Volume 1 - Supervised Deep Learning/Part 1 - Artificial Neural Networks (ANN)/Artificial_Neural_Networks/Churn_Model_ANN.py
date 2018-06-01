# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:06:49 2018

@author: Amit Gupta
"""
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#DATA PRE-PROCESSING
#------------------------------------------------------------------------------
#Importing Data 
Data = pd.read_csv(r"Churn_Modelling.csv")

#Triming our data to usefull Features.Removing Index No., Surname, CustomerID. 
#Irrelevant to the outcome of Model (i.e. Customer will Stay or leave)
X = Data.iloc[:,3:13].values
#Labels which we have. (Supervised Learning)
Y = Data.iloc[:,-1].values

#Encoding Categorical Vairables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelEnconder_X = LabelEncoder()
#Transforming different type to labels
X[:,1] = labelEnconder_X.fit_transform(X[:,1])

X[:,2] = labelEnconder_X.fit_transform(X[:,2])
#Since Data is not Oridnal.We Create Dummy variables
oneHotEncode_X = OneHotEncoder(categorical_features=[1])
X = oneHotEncode_X.fit_transform(X).toarray()
#Removing First Dummy Variable to avoid Dummy Variable Trap
X = X[:,1:]

#Spliting the Data into Train and Test Sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#------------------------------------------------------------------------------


#BUILDING ANN 
#------------------------------------------------------------------------------
import keras



