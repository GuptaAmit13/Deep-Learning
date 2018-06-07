# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 09:45:58 2018

@author: Amit Gupta
"""

# Recurrent Neural Network

#Data Pre-Processing
#%%
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Collecting Data
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
X = dataset_train.iloc[:, 1:2].values

#Scaling the Feature
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X_Scaled = sc.fit_transform(X)


#Creating Timesteps and Output
#%%

#X_Train : Last 60 Days Input
#Y_Train : last 60 Days Output

X_train = []
y_train = []
#Starting at Index 60th
for i in range(60, 1258):
    X_train.append(X_Scaled[i-60:i, 0])
    y_train.append(X_Scaled[i, 0])

#Converting to Array
X_train,y_train = np.array(X_train), np.array(y_train)

#Reshaping the array (Adding more Indicators)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


#Building RNN
#%%

#Importing Libraries for RNN
from keras.models import Sequential
#from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense

#Initializing RNN
regressor = Sequential()

#Adding LSTM Layer
regressor.add(LSTM(units=256,return_sequences=True,input_shape =(X_train.shape[1], 1),dropout=0.15))

#Adding more Layer
regressor.add(LSTM(units=256,return_sequences=True,dropout=0.15))
regressor.add(LSTM(units=256,return_sequences=True,dropout=0.15))
regressor.add(LSTM(units=256,return_sequences=True,dropout=0.15))
regressor.add(LSTM(units=256,return_sequences=True,dropout=0.15))
regressor.add(LSTM(units=256,dropout=0.15))

#Adding Output
regressor.add(Dense(units=1))


#Compiling RNN (Wil take Few Minutes)
#%%
#Compiling
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')

#Fitting Data to RNN Model
regressor.fit(X_train,y_train,epochs=125,batch_size=64)


#Predicting Price with RNN
#%%

#Ground truth Values
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
Y_real = dataset_test.iloc[:, 1:2].values

#Predicted Values axis : 0 vertical, 1 Horizontal
dataset = pd.concat((dataset_train['Open'],dataset_test['Open']),axis= 0)
test_inputs = dataset[len(dataset_train)-60:].values
test_inputs = test_inputs.reshape(-1,1)

#Scaling Input (Important). Fitting should be same as model trained on. Therefore, transform directly.
test_inputs = sc.transform(test_inputs)

X_test = []
#Starting at Index 60th
for i in range(60,test_inputs.size):
    X_test.append(test_inputs[i-60:i, 0]) 
X_test = np.array(X_test)

#Reshaping 
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

#Predicting
Y_pred = regressor.predict(X_test)
Y_pred = sc.inverse_transform(Y_pred)

#Visualizing Real VS Prediction
#%%

plt.plot(Y_real,color='red',label = 'Real Price')
plt.plot(Y_pred,color='blue',label = 'Predicted Price')
plt.title('RNN Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

