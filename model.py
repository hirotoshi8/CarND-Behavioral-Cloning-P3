# -*- coding: utf-8 -*-
"""
 Training the model from unity simulatpr
 
"""

import csv
from PIL import Image
import cv2
import numpy as np
"""
 Helper Function
"""
def def_process_image(source_path):
    """ Function: Get Image file as np.array
        Retunr: image file as np.array"""
    filename = source_path.split("/")[-1]
    current_path = "../CarND-Behavioral-Cloning-P3-Dataset/udacity_data/IMG/" + filename # for my data
    image = np.array(Image.open(current_path))
    return image


"""
 Reading the train data images and steering
"""
lines = []
with open(r"../CarND-Behavioral-Cloning-P3-Dataset/udacity_data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for csvline in reader:
        lines.append(csvline)
        

images = []
measurements = []
"""
 Data augumentation
"""
"""
 Mearge Center, Left and Right position of camera images
"""
# Work directory
directory = r"../CarND-Behavioral-Cloning-P3-Dataset/udacity_data/IMG"
for line in lines:
#    for count_i in range(3):
    steering_center = float(line[3])
    # create adjusted steering measurements for the side camera images
    correction = 0.2
    steering_left = steering_center + correction
    steering_right= steering_center - correction
    # read in images from center, left and right cameras
    img_center = def_process_image(line[0])
    img_left   = def_process_image(line[1])
    img_right  = def_process_image(line[2])

    # add images and angles to data set
    images.append(img_center)
    images.append(img_left)
    images.append(img_right)
    
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)

"""
### Flip the mirror images ###
"""
augumented_images = []
augumented_measurements = []
for image, measurement in zip(images, measurements):
    augumented_images.append(image)
    augumented_measurements.append(measurement)
    augumented_images.append(cv2.flip(image,1))
    augumented_measurements.append(measurement*(-1.0))

X_train = np.array(augumented_images)
Y_train = np.array(augumented_measurements)
#print(np.ndim(images[0]))
#print(X_train.shape)
#print(Y_train.shape)

"""
 Training process
"""
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

"""
### Create Model with NVIDIA's Architecture ###
"""
# model output
nb_classes = 1 # Steering
# Init Model
model = Sequential()
# Normalization
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape = (160,320,3)))

# Clopping
model.add(Cropping2D(cropping=((75,25),(0,0))))

# Create CNN
model.add(Convolution2D(24,5,5,subsample=(1,1),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(1,1),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(1,1),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())
#model.add(Dense(100, activation="relu"))
model.add(Dense(100))
#model.add(Dropout(0.5))
#model.add(Dense(50, activation="relu"))
model.add(Dense(50))
#model.add(Dropout(0.5))
#model.add(Dense(10, activation="relu"))
model.add(Dense(10))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))

"""
### Tuning Model Parameter ###
"""
model.compile(loss="mse", optimizer="adam")
"""
### Traing The Model ###
"""
model.fit(X_train,Y_train,batch_size=32,validation_split=0.2,shuffle=True,nb_epoch=3)
"""
 Save the Model
"""
model.save("model.h5")

