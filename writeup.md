#**Behavioral Cloning** 

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./output_images/center_2016_12_01_13_30_48_287.jpg "Normal Image"
[image6]: ./output_images/camera_image_sample.jpg "Normal Image"
[image7]: ./output_images/flipped_image_sample.jpg "Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

**My Behavioral Cloning Project Ooutput**
### I create the project output as video file [Here](./run_release.mp4). 
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used NVIDIA's neural network architechture.
My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 ( `Create Model with NVIDIA's Architecture` section in `model.py`)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

At first, I think the model has to contain dropout layers in order to reduce overfitting. However, from the resut of test run through the simulator, the behovour was more stable without dropout layer. This is why my model don't have the dropout layer. (`Create Model with NVIDIA's Architecture` section in `model.py`). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`Model Parameter Tuning` section in `model.py`).

#### 4. Appropriate training data

I used the data set provided by udacuty. This training data seems to be chosen to keep the vehicle driving on the road. 
For increasing the data set, I introduced some techniques.

- a combination of center, left and right camera images
- fliped the images to make the mirror images

Addition to that, I clipped the image for extracting only road area.

For details about how I created the training data, see the next section. 



## Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train model until over fitting and then reduce the validation accuracy and test run with simulator.

My first step was to use a convolution neural network model similar to the NVIDIA's architecture. I thought this model might be appropriate because they used this model for autonomous driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

In this project, only if used udacuty's dataset, it's not neccessary to combat the overfitting. So, I don't modify the model to add the Dropout layers.

The final step was to run the simulator to see how well the car was driving around track one.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`Create Model with NVIDIA's Architecture` in `model.py`) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	   	| 
|:---------------------:|:---------------------:| 
| Input         		| 320x160x3 RGB image   | 
| Convolution      	    | 5x5x24                |
| RELU				    | -		                |
| Convolution      	    | 5x5x36                |
| RELU				    | -		                |
| Convolution      	    | 5x5x48                |
| RELU				    | -		                |
| Convolution      	    | 3x3x64                |
| RELU				    | -		                |
| Convolution      	    | 3x3x64                |
| RELU				    | -	                	|
| Fully connected(fc1)	| 100		            |
| Fully connected(fc2)	| 50		            |
| Fully connected(fc3)	| 10		            |
| Fully connected(fc4)	| output = 1 (The number of unique classes in the data set)|


#### 3. Creation of the Training Set & Training Process

In this project, I used udacity's data set.
Here is an example image of lane driving:

![alt text][image2]

To capture good driving behavior, data augumentation is important. In this step, I increased the data set with some techniques.

- a combination of center, right and left camera
- flipe the image

### A combination of center, right and left camera image

In this project, simulator has three cameras at center, left, right side of the car.

![alt text][image6]

However, I didn't have the steering data for each camera positions. Due to lack of the label data (steering data), I create the label data based on the center camera's steering data below. (Code is in `Combination Center, Left and Right position of camera images` in `model.py`)

```sh
steering_left = steering_center + correction
steering_right= steering_center - correction
```
In this formula, `correction` is the parameter to tune the degree of the steering.
(For this project, I set `correction` is `0.2`)

### Flip the image

To augment the data set, I also flipped images and angles thinking that this would icrease the `Right turn`(clockwise driving) data. Basically, this test course is counter-clockwise course and `Left turn` steering is much more `Right turn`. This is because the data set is unbalanced, it had better to create the mirror images. For example, here is an image that has then been flipped:

![alt text][image7]

## Pre-process
Before training process, my project has two preprocess step.

- clipt the image
- Normarization

### Clip the image
To train model, only road area is neccessary. This means that half of image is not neccessary. This is why I clipped the road area from the images.

### Normarization
After the data collection process, I had 38572 number of data points. I then preprocessed this data by Nomarization(mean:0 to 1, variance: -1 to +1).

## Train Model
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

### Model Parameter Tuning
 I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced experimentally. I used an adam optimizer so that manually training the learning rate wasn't necessary.

| Parameters       		|     Description	        			| 
|:---------------------:|:-------------------------------------:| 
| Optomizer        		| AdamOptimizer   						| 
| Batch Size         	| 32                                    |
| number of epochs		| 3					                    |
