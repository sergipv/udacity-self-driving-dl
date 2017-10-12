# Traffic Sign Recognition

## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[image1]: ./examples/traffic_bunch.png "Bunch of traffic images"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new_images/stop.jpg "Stop"
[image5]: ./new_images/crossing.jpg "Children Crossing"
[image6]: ./new_images/intersect.jpg "Intersect"
[image7]: ./new_images/end_speed.jpg "End speed"
[image8]: ./new_images/roundabout.jpg "Roundabout"
[image9]: ./new_images/work.jpg "Work ahead"
[image10]: ./new_images/speed_130.jpg "Max Speed 130km/h"

### Loading datasets

Before starting to work on the classificaiton itself, it is good to explore and visualize what images are part of the dataset we are trying to classify and what is the distribution of the different classes. I was using 3 pickle files that contain features and labels for training, validation and test datasets. The initial features are the images (32x32x2 pixel images) from the traffic signs.

The three datasets have sizes of 34799 (training), 4410 (validation) and 12630 (test).

There are a total of 43 classes/labels that we want to classify to, corresponding a different types of traffic signs. The distribution of the classes is not uniform, which might make our models overfit for these classes for which we have more data (Speed limit of 20 and 50km/h are the top 2 classes). While the accuracy of a system might be higher for a specific model if there are more samples of these classes we have more training data, we probably want a system that is able to classify correctly independently of the class of the image. While I didn't do any specific work on that, I think that adding accuracy levels per class would be useful to determine if one (or more!) of our models is overfitting specific classes. This lack of a uniform distribution per class is not unique to the training data: test and validation data are also non-uniform. We can also augment the data for these classes that lack enough data to avoid unbalance classification.


### Exploratory visualization of the dataset.

It is usually a good idea to visualize some of the images from the dataset:

![alt text][image1]

The images on the dataset have been taken with different illumination and angles. While data augmentation might not be as useful as when we have images with the same saturation, position, etc it should still be a good way to provide more data to our system (maybe even trying to balance the class distribution). From the 3 models that I implemented, I used some data augmentation in the batch_normalization model. In this model, while I didn't augment the data in the pre-processing step, I modified randomly the brightness and the contrast before training. This helped increasing the accuracy for this model (from 0.88 aprox up to > 0.93); since we're running multiple epochs, this pre-processing can be seen as a data augmentation step. I didn't want to spend an excessive amount of time in this step, but if I were to add a more fine data augmentation, I would have focused on augmenting classes that are underrepresented in the training set.


### Design and Test a Model Architecture

#### 1. Data pre-processing

After visualizing some images, it was pretty clear that the features (images) were not normalized, so I proceeded to normalize the images. While I considered implementing some data augmentation for the training, after some tests generating images with random changes in brightness and contrast, they did not provide a sufficient improvmenent in final accuracy (even though I didn't do a extense work in that area). I did not consider flipping the images since the traffic signals would be different in some cases (right turn vs left turn for instance).

As noted before, if I were to spend more time on data augmentation, I would focus on trying to generate more images for these classes that have lower number of training samples.


#### 2. Model Architecture

The final model I implemented is an ensemble of three diferent models: leNet, a leNet-based model with batch normalization and a model based on VGG. All models were trained separately, with the ensemble combining the results on evaluation time. In general, I did not keep the model with highest validation during training, but the last model I got; ie: if I trained for 20 epochs and the model had higher validation accuracy at epoch 18 than at epoch 20, I would still take the model at epoch 20 (mostly for simplicity).

The ensemble was implemented by evaluating the three models separately and adding the scores.

- LeNet-based:

This was the fastest model to train; it also got a pretty good validation (0.959) and test accuracy values (0.946). The main difference with LeNet was the usage of a dropout layer, where we drop weights on the network with a probability of 0.5. This model was also the faster to converge into a model with acceptable results.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				|
| Flatten		| 1x4096
| Fully connected		| 256
| RELU					|		|
| Dropout		| 0.5 probability of dropping o weight 		|
| Fully connected		| 128
| RELU					|		|
| Dropout		| 0.5 probability of dropping o weight 		|
| Output 	| 43 		|
 

- Batch Normalization:

The batch normalization model was not getting initial good results (accuracy below 0.9), so I added a few extra preprocessing steps before training the image: the images were modified with a random brightness, random constrast and the final images were made to grayscale as a preprocessing step before training. This allowed an increase on the validation (0.934) and test accuracy (0.932). 

The batch normalization model was:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Grayscale		| 32x32x1 grayscale image
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Batch Normalization	| 8x8x64
| Batch Normalization	| 4x4x128
| Dropout		| 0.5 probability of dropping o weight 		|
| Reshape		| 4x4x128	|
| Fully connected		| 128
| Dropout		| 0.5 probability of dropping o weight 		|
| Output 	| 43 		|
 

- VGG-based:

The final model is deeper than the rest, but not as deep as a VGG-16. While implementing the model, anything higher than 3 convolutional layers (each containing two layers) was not training correctly the data. While I am not completely sure of the reasons, I have the feeling that networks to learn 32x32x3 images cannot be too deep, or at some point the features are lost, and the network can't train. I would like to try the same model with higher dimension images and test the best depth to test these images.

This VGG-based model ended up having the higher accuracy level from the three independent models, but took longer to train per epoch and also needed more epochs to plateau.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x128	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x256 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x256	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x256 				|
| Flatten		| 1x4096
| Fully connected		| 1024
| RELU					|		|
| Dropout		| 0.5 probability of dropping o weight 		|
| Fully connected		| 512
| RELU					|		|
| Dropout		| 0.5 probability of dropping o weight 		|
| Fully connected		| 128
| RELU					|		|
| Dropout		| 0.5 probability of dropping o weight 		|
| Output 	| 43 		|
 



#### 3. Training the models.

As I mentioned, I trained three different models and I combine them in an ensemble. The structure of training is similar for each model: all of them use Adam optimizer, with a learning rate of 0.001. The cost function to minimize was the mean of tf.nn.softmax_cross_entropy_with_logits for all models.

I ended up erring a little bit on the longer side of number of epochs (since that gave the final ensemble a bit boost of 0.1-0.2 boost). The number of epochs I used was 25, 50 and 30 forLeNet, Batch Normalization and VGG. The main reason why I added way more epochs to the Batch Normalization model was the random pre-processing of the images before the training.


#### 4. Results

Each of the models have a validation and test accuracy > 0.93. For each individual model:

LeNet:
* training accuracy of 0.959
* validation accuracy of 0.959
* test accuracy of 0.953

BatchNormalization:
* training accuracy of 0.934
* validation accuracy of 0.939
* test accuracy of 0.935

VGG-based:
* training accuracy of 0.970
* validation accuracy of 0.970
* test accuracy of 0.960


The final ensemble outperforms all individual models, being able to increase validation and test accuracy:
* validation accuracy of 0.979
* test accuracy of 0.975


The first architecture I trained was based on LeNet, since it had good results for similar size images. This allowed to start with a similar structure for strides, number of convolutional and fully connected layers, and with a similar number of dimensions in each step. The main difference with the numbers database was that the current database had 3 color layers instead of a grayscale, but after the first implementation the results were towards the good direction (>0.88). Adding dropout in the fully connected layers bumped the accuracy levels to the area we were looking (validation > 0.93).

For the second architecture, I used a similar structure than LeNet, but adding Batch Normalization instead of non-normalized convolutional layers. Starting with the same architecture than LeNet and iterating, allowed doing more testing on this. At the same time, since the final model I wanted to use was an ensemble, I didn't want to copy the structure and just adding normalized batches (I had some fear of both models learning the same features and the ensemble not adding much). At the end, doing some preprocessing with random brightness and random contrast allowed the model to reach the desired levels (>0.93). At the end, though, the results with the ensemble with this two models was too similar (so even though I tried to "force" to learn a bit different, I couldn't make it happen). Increasing the number of iterations helped boosting a bit more the ensemble so I decided to add more epochs.

The final architecture was based on VGG. In this architecture, we have a very deep network, with up to 16 convolutional plus fully connected layers. In this case, I ended up removing a lot of convolutional layers since the model stopped learning after 6 convolutional layers (with the strides I used, at least). The final model was the best of the individual models; in some cases it reached validation accuracy > 0.97.

In general, the use of dropout was needed to avoid overfitting. Clearly the networks generalize pretty well, since training, validation and test accuracies are in the same range. I would have been worried if, for example, training accuracy was on a high (>0.93) value, but validation and/or testing in a lower value (for instance, lower than 0.8).

The ensemble of all three models ended up boosting the final test accuracy quite a bit (0.96 to 0.975 in the last training I did, even though in some cases it reached 0.988 test accuracy). As I mention before I wanted to try ensemble with different architectures to see if they could provide higher accuracy when combining their learnings, which ended up being the case. The main challenge was mostly technical, since I wasn't aware of the tf.variable_scope and it took a bit until I got this piece working.


### Test a Model on New Images

#### 1. Select some new images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10]

I would consider most images easy to classify; the only two that could be a bit more problematic were the "Childred crossing" and the "Work ahead" signs, since they aspect ratio of the signs is not the correct one. While I did some testing with some darker images, I didn't see any difference on the classifer not being able to classify them. In general, the classification was accurate even for modified images. I also tried to choose images from labels that didn't have a lot of training data proportionally to the rest.

While I would have loved to test more, one of the problems I had was to find images without watermarks that I could use. I ended up cropping images from larger images, which might not be ideal for this case.

One thing that I was interested on was what was the model going to predict with signs that the model wasn't trained on. For this specific case I added a sign with no training data, and I hoped the ensemble would have classified the sign as the closest from the training labels. In this case, the image was a top speed at 130km/h, so I hoped the top score was 120km/h, but strangely, 120km/h was not even in the top 5, and the ensemble was quite sure the image was a "End of no passing". The only explanation I could find was that one of the sides of the road seems the 45 degree gray line from a "End of no passing" traffic sign. In any case, it made me wonder how to get more generalizable results for non-seen images.

#### 2. Results

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| End of all speed and passing limits		|	End of all speed and passing limits 		|
| Right-of-way at the next intersection		|	Right-of-way at the next intersection		|
| Road Work			|	Road Work		|
| Roundabout mandatory		| 	Roundabout mandatory 		|
| Stop		| Stop		|
| Children Crossing		| Children Crossing		|
| 130kmh (not available)	| End of no passing		|


The model was able to correctly guess 6 of the 6 traffic signs for which we had labels, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set, but this can be given by the limited amount of data I used compared to the test set.

#### 3. Comparing top-5 results for the test images

The usage of the ensemble makes it pretty clear for all images which is the correct image. In no case there was much confusion and all the images were classified with a 1.0 score (the score is not really 1.0, but in the 3rd order of magniture closer to 1 (>0.999). Same thing for the other 4 top results.


