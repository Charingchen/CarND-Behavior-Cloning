# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/pilotNet.PNG "Model Visualization"
[image2]: ./images/nn.svg "Model Visualization"
[image3]: ./images/center.jpg "Model Visualization"
[image4]: ./images/recover.gif "Model Visualization"
[image5]: ./images/recover2.gif "Model Visualization"
[image6]: ./images/Second_Track.gif "Model Visualization"


## Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
A modified [PilotNet](https://arxiv.org/pdf/1704.07911.pdf) was introduced. The modified decisions are explained below in the next section. The code is located between line 73 to 94 in model.py.  

#### 2. Attempts to reduce overfitting in the model

The model contains three dropout layers in order to reduce overfitting (model.py lines 88,90,92). All of them are located after dense layers and have a dropout rate of 50%.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 97).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I have one lap driving around the centre of the road. And I also include recover data driving from the side of the road and steer toward the centre.
For details about how I created the training data, see the next section. 

### Detailed Design and Approach

#### 1. Solution Design Approach
First I used original un-changed PilotNet architecture as shown below. ![alt text][image1]

I used Keras cropping 2D layers to crop out only the road image before feeding into the Convolution layers. The sky and hood of the car are not going to useful to determine the steering angle, therefore I removed those from the trainning data. Then, I applied a Lambda layer to normalize the image by dividing all channels to 255 and minus 0.5 to center around 0.

In term of the activation function, I found a paper talking about [exponential linear unit (ELU)](http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf). Instead of using RELus, this ELU would avoid a vanishing gradient via the identity for positive values. So I decided to use this instead of RELUS.

However, when I run this model and test in the simulator, the car would steer hard right and drove off the road with a determination. No correction behavior at all. 

Therefore, I included the left and right camera images into the training data set with a steering angle correction. Then, the trained model shows steering correction behavior and able to turn back to the center couple times. But it still shows it would turn right more than turn left.

Then I applied image mirroring to all the images I have to increase the  data set. After this, the model seems to have equivalent eight on right and left turns. 

During last couple training, I noticed the model often get overfitted after 3-4 epochs. I realized the model might be too complicated for the this. After all, this model is aimed to drive a car in real world. Therefore, I remove the 1164 neurons dense layers to reduce complicity. I was able to drop down the validation loss from 0.07 to 0.05 in first few epochs.

When the model train around 6,7 epochs, the validation loss start to become more than the training loss. I started to add dropout after each dense layers until I see the validation loss and training loss would be around the same at 7 epochs.

Once the training loss and validation loss drops around 0.05, the trained model was able to drive around the trace without steering off the track.

I also experiment changing the normalization method. Instead of dividing 255, i divided by 127.5 - 0.5. However, the model would perform poorly when there is a texture change on the side of the road.


I also tried to use generator showed in the class to hold the image data. But, it would slow down the training significantly, since with the version of tensorflow and keras in the VM, I am still able to store all image in the memory. I decided not use generator. With Tensorflow version 2 and latest Keras, model fit would take a Sequence object to achieve similar result.

 

#### 2. Final Model Architecture

The final model architecture consisted of three convolution layers with kernal size of 5x5 and two convolution layers with kernal size of 3x3. And then followed by 100,10,1 dense layer to classify the final result.

Here is a visualization of the architecture 

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving and one lap reversed. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to hard recover when it is at side of the track.

![alt text][image4]![alt text][image5]

Then I repeated this process on track two in order to get more data points.

![alt text][image6]

To augment the data sat, I also flipped every images and multiply the steering to -1 to gain 2x data. 

After the collection process, I had 4852x3 number of images. and after the flipping the images, I have 29112 images to feed into the model.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7. After 7 epochs, the vaildation loss would increase from 0.05 to 0.06 to 0.7 sometime. I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Udacity Review Feedback
### Model Architecture and Training Strategy
You used convolutional layers with input normalized in the Lambda layer. Exponential Linear Units were built into the architecture of the neural network as a nonlinear activation function.

Resources on activation methods:

Commonly Used Activation Functions
http://cs231n.github.io/neural-networks-1/#actfun

### Learning Rate
The [Adam optimiser](https://arxiv.org/pdf/1412.6980v8.pdf) was used as an [adaptive learning rate method](https://cs231n.github.io/neural-networks-3/#ada).

### Architecture and Training Documentation
You described the process that led you to the final implementation. Well done.

I recommend watching the following talk given by Andrej Karpathy at Deep Learning School in Stanford to revise the design of convolutional neural network architectures if needed: https://www.youtube.com/watch?v=u6aEYuemt0M

Optional video about training neural networks (and much more):

Nuts and Bolts of Applying Deep Learning (Andrew Ng)
https://www.youtube.com/watch?v=F1ka6a13S9I

### Additional Resources
[How Udacity’s Self-Driving Car Students Approach Behavioral Cloning](https://medium.com/udacity/how-udacitys-self-driving-car-students-approach-behavioral-cloning-5ffbfd2979e5)

[Self-driving car in a simulator with a tiny neural network](https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234#.2ugw8o56l)

---
## Dependencies

This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.
