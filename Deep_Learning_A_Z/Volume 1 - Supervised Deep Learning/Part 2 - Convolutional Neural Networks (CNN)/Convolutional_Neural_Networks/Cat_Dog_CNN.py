# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:54:38 2018

@author: Amit Gupta
"""
#Building CNN for CATS AND DOGS
#%%
#Importing Libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#%%
#Creating CNN Layers and Compiling

#Initializing CNN
classifier = Sequential()

#First Layer - Convolution 

#Convolution(#no of convolutions(feature detectors),#no of rows,#no of cols,inputshape(dimesion of 2d array,no of channel))
#If using Tensorflow Backend
classifier.add(Conv2D(32,(3,3),input_shape = (128,128,3),activation='relu'))

#If using Theano Backend
#classifier.add(Convolution2D(32,3,3,input_shape = (3,64,64),activation='relu'))

#Pooling Layer
classifier.add(MaxPool2D(pool_size=(2,2)))

#Adding Another Convolution Layer 
classifier.add(Conv2D(32,(3,3),input_shape = (32,32,3),activation='relu'))

#Another Pooling Layer
classifier.add(MaxPool2D(pool_size=(2,2)))

#Falttening Layer
classifier.add(Flatten())

# Full Connection  Layers (ANN)
#Hidden Layer
classifier.add(Dense(units = 256,activation='relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 128,activation='relu'))
classifier.add(Dropout(rate=0.2))

#Output Layer
# Binary : output_dim = 1 otherwise no. of classes  
classifier.add(Dense(units= 1,activation='sigmoid'))

#Compiling 
classifier.compile(optimizer='adam',metrics=['accuracy'], loss='binary_crossentropy')

#%%
from keras.preprocessing.image import ImageDataGenerator

#Image Augmentation (To reduce Overfitting and enrich dataset)
#Code taken from Keras Documentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128,128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

#This step may take few mins to hours depending upon the hardware
classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)