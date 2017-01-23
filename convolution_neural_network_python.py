# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 23:37:51 2017

@author: AmoolyaD
"""

#Building the CNN
#Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Creating the classifier
classifier = Sequential()

#Step 1 Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (3, 64, 64), activation= 'relu'))

# Step 2 Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 Flattening
classifier.add(Flatten())

#Step 4 Full Connection
classifier.add(Dense(output_dim=128, activation= 'relu'))
classifier.add(Dense(output_dim=1, activation= 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

