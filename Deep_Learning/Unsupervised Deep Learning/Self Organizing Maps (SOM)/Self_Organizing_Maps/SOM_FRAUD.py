# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:23:30 2018

@author: Amit Gupta
"""
#Data Pre-Processing 
#%%
#Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Collecting Data
data = pd.read_csv(r"Credit_Card_Applications.csv")
X = data.iloc[:,:-1].values
#Just to differ customer who got approved or not. Not for learning.
Y = data.iloc[:,-1].values

#Feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X_scaled = sc.fit_transform(X)



#Learning/Train the Model
#%%
from minisom import MiniSom

#X,Y : Dimension of SOM
som = MiniSom(10,10,input_len=X.shape[1],sigma=1.0,learning_rate=0.001)

#To Initialize Random weights
som.random_weights_init(X_scaled)

#To Train Model
som.train_random(data=X_scaled,num_iteration=10000)

#Visualizing the results
#%%

from pylab import bone, pcolor, colorbar,  plot ,show
bone()
pcolor(som.distance_map().T)
#Legend
colorbar()
#Customer Marker Red: Not Approved Green:Aprroved
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X_scaled):
    #Winner Node for That Customer
    w = som.winner(x)
    #Co-ordinates of Winner Node. Middle point
    plot(w[0] + 0.5,w[1] + 0.5,markers[Y[i]],markeredgecolor = colors[Y[i]],markerfacecolor='None',markersize=10,markeredgewidth =2 )
show()
   

#Finding Frauds 
#%%
mappings = som.win_map(X_scaled)
frauds = np.concatenate((mappings[(5,1)],mappings[(5,3)],mappings[(6,1)],mappings[(7,1)]),axis=0)
frauds = sc.inverse_transform(frauds)