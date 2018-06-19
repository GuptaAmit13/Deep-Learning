# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:23:35 2018

@author: Amit Gupta
"""
#Importing Libraries
#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


#Collecting Dataset
#%%
movies = pd.read_csv(r'ml-1m/movies.dat',sep= '::',header=None,engine= 'python',encoding='latin-1')
users = pd.read_csv(r'ml-1m/users.dat',sep= '::',header=None,engine= 'python',encoding='latin-1')
ratings = pd.read_csv(r'ml-1m/ratings.dat',sep= '::',header=None,engine= 'python',encoding='latin-1')


#Separating data (Train,Test)
#%%
Training_set = pd.read_csv(r'ml-100k/u1.base',delimiter ='\t')
Training_set = np.array(Training_set,dtype = 'int')
Test_set = pd.read_csv(r'ml-100k/u1.test',delimiter ='\t')
Test_set = np.array(Test_set,dtype='int')


#Creating a User vs Movie rating matrix
#%%
no_users =int( max(max(Training_set[:,0]),max(Test_set[:,0])))
no_movies = int( max(max(Training_set[:,1]),max(Test_set[:,1])))

#Function to Convert
def convert(data):
    converted_data = []
    for user in range(1,no_users+1):
        id_movies = data[:,1][data[:,0] == user]
        id_ratings = data[:,2][data[:,0] == user]
        user_ratings = np.zeros(no_movies)
        user_ratings[id_movies-1] = id_ratings
        converted_data.append(list(user_ratings))
    return converted_data

#Converted Matrix
Training_set = convert(Training_set)
Test_set = convert(Test_set)

#Converting to Torch Tensors(Vectors)
#%%
Training_set = torch.FloatTensor(Training_set)
Test_set = torch.FloatTensor(Test_set)

#Converting ratings to Binary (Yes:1 or No:0)
#%%
# -1 for Movies not Rated by Users
Training_set[Training_set == 0] = -1

# 1 for Movies Liked by Users
Training_set[Training_set >= 3] = 1

# 0 for Movies Not Liked by Users
Training_set[Training_set == 1] = 0
Training_set[Training_set == 2] = 0

# -1 for Movies not Rated by Users
Test_set[Test_set == 0] = -1

# 1 for Movies Liked by Users
Test_set[Test_set >= 3] = 1

# 0 for Movies Not Liked by Users
Test_set[Test_set == 1] = 0
Test_set[Test_set == 2] = 0


#Restricted Boltzman Machine
#%%
class RBM():
    def __init__(self,no_of_visible,no_of_hidden):
        self.Weights = torch.randn(no_of_visible,no_of_hidden)
        # 2-d tensor. 1st for Batch, 2nd Bias
        self.Bias_hidden = torch.randn(1,no_of_hidden)
        # 2-d tensor. 1st for Batch, 2nd Bias
        self.Bias_visible = torch.randn(1,no_of_visible)
        
    def sample_h(self,x):
        Weights_x = torch.mm(x,self.Weights.t())
        activation = Weights_x + self.Bias_hidden.expand_as(Weights_x)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    