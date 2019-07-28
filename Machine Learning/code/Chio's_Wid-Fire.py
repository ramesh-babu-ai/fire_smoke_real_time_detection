# %% [code]
from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import zipfile


import zipfile

Dataset = "Wildfire Photos"
print(tf.__version__)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# %% [code]

import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
import numpy as np
from time import time
from time import sleep
import random
import cv2

Wildfire_folder = "../input/data/wildfire"
Non_wildfire_folder = "../input/data/non-wildfire"
wildfire = [f for f in os.listdir(Wildfire_folder) if os.path.isfile(os.path.join(Wildfire_folder, f))]
non_wildfire = [f for f in os.listdir(Non_wildfire_folder) if os.path.isfile(os.path.join(Non_wildfire_folder, f))]
im = []
im5 = []
label = []
test_data = []
test_label =[]
for img in wildfire[:20]:
    temp = cv2.imread("../input/data/wildfire/"+img)
    
    if temp is not None:
        temp = cv2.resize(temp,(256,256),interpolation=cv2.INTER_LINEAR)
        im.append(temp)
        label.append(1)
for img in non_wildfire[:20]:
    temp = cv2.imread("../input/data/non-wildfire/"+img)
    if temp is not None:
        temp = cv2.resize(temp,(256,256),interpolation=cv2.INTER_LINEAR)
        im.append(temp)
        label.append(0)

print(len(im))








# %% [code]
print(len(im))
for a in range (0,10):
    random_num = random.randint(0, len(im)-1)
    print (np.asarray(im[random_num]).shape)
    test_data.append(im.pop(random_num))
    test_label.append(label.pop(random_num))
print (len(test_data))
print(len(im))

im = np.asarray(im)
test_data = np.asarray(test_data)

# %% [code]
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add (Conv2D (filters=96, input_shape=(256,256,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer1
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(1))
model.add(Activation('softmax'))

#model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

label = np.array (label)

# %% [code]
from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator (
    rescale = 1./255,
    shear_range =0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator ( rescale = 1./255)

training_set = train_datagen.flow_from_directory (
    '../input/data/',
    target_size =(256,256),
    batch_size =5,
    class_mode = 'binary'
)



# %% [code]
model.fit_generator (
    training_set,
    steps_per_epoch = 8000,
    epochs =10
    
)