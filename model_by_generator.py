"""
 Training process
"""
import csv
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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
path = "../CarND-Behavioral-Cloning-P3-Dataset/udacity_data/driving_log.csv"
# Work directory
directory = r"../CarND-Behavioral-Cloning-P3-Dataset/udacity_data/IMG"    
lines = []
with open(path) as csvfile:
    reader = csv.reader(csvfile)
    for csvline in reader:
        lines.append(csvline)
samples = lines
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


"""
 Data generator augumentation
"""
# create adjusted steering measurements for the side camera images
correction = 0.2
center_angle=0
left_angle  = center_angle + correction
right_angle = center_angle - correction
    
def generator(samples, batch_size=32):
    """ Function: Mearge Center, Left and Right position of camera images
        Retunr  : Merged data """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:(offset+batch_size)]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Data augumentation for images
                center_image = def_process_image(batch_sample[0])
                left_image   = def_process_image(batch_sample[1])
                right_image  = def_process_image(batch_sample[2])
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                # # Data augumentation for Steering
                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            # Set the training data
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

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
model.add(Convolution2D(24,4,4,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,4,4,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,4,4,subsample=(2,2),activation="relu"))
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
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)
"""
 Save the Model
"""
model.save("model_by_generator.h5")

"""
 Print learning history
"""
# Print the keys contained the history object
print(history_object.history.keys())
# Plot the training and validation loss for each eopchs
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squeared error loss')
plt.ylabel('mean squeared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()