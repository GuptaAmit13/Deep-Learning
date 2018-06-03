# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:06:49 2018

@author: Amit Gupta
"""
#%%

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
#%%

#BUILDING ANN 

#-----RUN THIS PART FOR SIMPLE ANN---------------------------------------------
#Importing Libraries for building ANN
import keras
from keras.models import Sequential
from keras.layers import Dense


#Creating ANN Model
#Initializaing ANN
classifier = Sequential()
#Input layer and First Hidden Layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim = 11))
#Adding Another Hidden Layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
#Adding Output Layer
# for Binary classification use 'sigmoid' activation, for multiclass classfication use 'softmax' activation and change units to no. of expected output class
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#Compiling ANN
#Use 'categorical_entropy' for multiple classes
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting ANN to Dataset
classifier.fit(X_train,Y_train,epochs=150,batch_size=10)

#Predicting the Test set
#Gives Probability
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
#------------------------------------------------------------------------------


#Tuning  and Improving ANN
#------RUN THIS PART FOR TUNED AND IMPROVED ANN--------------------------------

#Libraries for This Part
#%%
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout


#Bias-Variance Trade-off
#To check we are getting the optimal accuracy.
#%%
#Evaluating ANN Model

def build_classifier():
    #Creating ANN Model
    #Initializaing ANN
    classifier = Sequential()
    #Input layer and First Hidden Layer
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim = 11))
    #Adding Another Hidden Layer
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    #Adding Output Layer
    # for Binary classification use 'sigmoid' activation, for multiclass classfication use 'softmax' activation and change units to no. of expected output class
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    #Use 'categorical_entropy' for multiple classes
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,epochs=150,batch_size=10)

#Applying K-Fold Cross Validation Method for Better Accuracy. In this case, K=10
accuracies = cross_val_score(estimator=classifier,X = X_train, y = Y_train, cv = 10,n_jobs=1)
mean = accuracies.mean()
variance = accuracies.std()

#%%
#Improving the ANN

def build_classifier(optimizer):
    #Creating ANN Model
    #Initializaing ANN
    classifier = Sequential()
    #Input layer and First Hidden Layer with dropouts
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim = 11))
    classifier.add(Dropout(rate= 0.1))
    #Adding Another Hidden Layer with dropout
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dropout(rate= 0.1))
    #Adding Output Layer
    # for Binary classification use 'sigmoid' activation, for multiclass classfication use 'softmax' activation and change units to no. of expected output class
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    #Use 'categorical_entropy' for multiple classes
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn = build_classifier)

#Hyper Parameter Dictionary
parameters = {'batch_size':[20,22,24,26,28,30,32,34,36,38,40,42,44,46],
              'epochs':[100,500,1000],
              'optimizer':['adam','rmsprop']} 

#Using GridsearchCV
grid_search = GridSearchCV(estimator=classifier,param_grid = parameters,scoring = 'accuracy',cv = 10).fit(X_train,Y_train)
best_parameter = grid_search.best_params_ 
best_accuracy = grid_search.best_score_