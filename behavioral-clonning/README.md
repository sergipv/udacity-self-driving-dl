# Behavioral Cloning

The goal of this project are:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Use the simulator to collect data of good driving behavior

There were some issues when I downloaded an old version of the simulator, which used as input discrete values from the keyboard, instead of using the mouse as input. The quality of the data obtained was bad (most steering angle equals to 0.0); after realizing that there were other simulators that recorded data from the mouse, I downloaded and used the second simulator, which greatly improved the training data.

Histogram of angles using the old simulator:

[image1]: ./images/old_simulator.png "Old Simulator"
![alt text][image1]

Histogram of angles using the new simulatori (before flipping images):

[image2]: ./images/collected_4.png "Old Simulator"
![alt text][image2]

After collecting this data, it became clear that there was some pre-processing needed: displaying a histogram of the steering angle data, showed that there most of the data was too close to 0.0 (which trains for no change in steering angle). Also, it became apparent that more data needed to be generated (augmented) to have more simmetry on the input data. After a few mostly-successful simulations, I also recorded more data in several areas where the autonomous mode was not doing great: the bridge, and the two areas where there is a separated road made of sand. The data I recorded was in both directions.

I followed a few more strategies during data collection: I recorded both directions of the circuit and after using the beta simulator, recorded some recovery paths (ie: drive outside of the correct path without recording data, and recovered to the center of the road while recording data). Finally I tried to record driving simulation data using different speeds to try to mimic the simulation speed (it became clear that the faster the car was driving in autonomous mode, the more difficult was the simulation).

Overall, data collection was a process I followed until the end of the project; increase training data is one of the main strategies I would follow to improve the autonomous driving.


## Pre-processing training data

I implemented two different generators, which ended up giving similar results. In both generators, not all images were accepted; if the image was associated with a steering angle smaller than a certain threshold (0.02 degrees) the image was discarded 94% of the times. While I chose 94%, I didn't see much differences between 92 to 98% filtered. The generators were as following:

* Generator 1: simpler generator. 
  * Loads all the training data (for left, center and right images).
  * Flips all training data and adds it to the training and test data.
  * The angles for the left images are corrected with +0.2 and the right images with a -0.2.
* Generator 2:
  * Randomly selects one timestamp from all timestamps available.
  * Randomly selects one of the three images (left, center or right) correcting the angle recording depending on the image using the same approach than the Generator 1. 
  * Changes the Value channel of the image after transforming the image to HSV space.
  * Flips the image 50% of the time.

I didn't add more pre-processing, but there are a few more that I would try to help generalizing the training data:
* Transform to grayscale.
* Shear operation.
* Slight rotation of images.

## Architecture used

The final model architecture I used was extracted from [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf), a paper from NVIDIA on their approach to self driving cars. I tested other models (starting with LeNet) but I could not get similar results and the NVIDIA model had significant smaller number of parameters compared to other models (~600K parameters vs up to 10M parameters):

* Cropping the image from (160,320) to (65, 320).
* Normalization to [0,1] range of the image values.
* Convolutional Layers with an "elu" activation function:
  * Convolutional layer with a depth of 24 and a 2x2 stride and a 5x5 kernel.
  * Convolutional layer with a depth of 36 and a 2x2 stride and a 5x5 kernel.
  * Convolutional layer with a depth of 48 and a 2x2 stride and a 5x5 kernel.
  * Convolutional layer with a depth of 64 and a 3x3 kernel.
  * Convolutional layer with a depth of 128 and a 3x3 kernel.
* Dropout with 0.5 probability.
* Flatten
* Dense layer to 100 neurons.
* Dense layer to 50 neurons.
* Dense layer to 10 neurons.
* Dense layer to 1 neurons.

The final output is the predicted angle. The optimizer used is an Adam optimizer with a learning rate of 0.0001 and the loss function used is mean square error.

As I mentioned, I tried other models; with LeNet model I got good training and validation loss but the results on the simulator were not good. I could improve the model by removing the last Convolutional layer (3 convolutional layers followed by max pooling) followed by 5 dense layers). Adding more layers seemed to make training more complex, and MaxPooling did not seem to improve the simulation results (even though training and validation loss improved!).

Overall this model and general strategy allowed full autonomous driving in the first circuit up to 25mph and some driving on the 2nd circuit (mostly needed more data to be able to navigate in the second circuit).

## Avoiding overfitting

There are different ways that the solution I chose fights overfitting. First, while data collection ended up being more focused on the areas that the autonomous driving was having trouble, I did recordings using different strategies (on and off of recording while recovering the car, for example. Also, I didn't use large number of epochs (5 to 10 epochs per dataet) for training the system and I trained & saved sequentially with multiple datasets. These datasets were also recorded on both directions, which greately helped the training process.

I also split the training data between test and validation data. The deep CNN used also had one dropout layer to avoid overfitting after the convolutional layers.

In any case, some type of overfitting seems necessary; to be able to make the car drive on the second circuit large amount of data for that second track was necessary. In an ideal world, the model should have been able to learn unseen data and make accurate predictions. But the strong differences in the circuit (for instance, the line separation in the second circuit) makes this generlization complex.


## Future work

Projects like this open possibilities of adding more strategies on the different building blocks of the system. There are a few things I would consider for improving the driving system (for instance, smooth driving on the second circuit,  reach higher speeds during simulation or improve generalization for other circuits).

* Data Augmentation:
  * The easier strategy is keep adding more training data, maybe using some other people to record the circuit in both directions. Also, use (even more) different velocities into the system, and maybe add the velocity as parameter to the network to improve prediction.
  * Usage of grayscale or other image processing techniques to add more training data.
  * Shear operation.

* Architecture:
  * I would like to try to reduce the number of parameters of the network as much as possible while still getting an acceptable driving capability.
  * At the same time, try to establish what is the limit of depth for a network to learn this task (and why).

There are other general strategies that I would have liked to test:
  * Don't use the precise steering angle for the current image, but apply a low pass filter with the last N predictions and use that value for the steering angle to try to smooth the driving.
  * Add more pre-processing or computer vision on the images that could improve training. One example could be some line finding (similar to the first or forth project) that would make lanes easier to train (probably I will revisit this project after finishing the forth project for improved lane detection)

