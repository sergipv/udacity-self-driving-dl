# Vehicle Detection

The goal of this project is to implement a pipeline that is able to detect and track vehicles. In this project, the goal is not to use the best image processing and deep learning classification system but to use other Machine Learning supervised systems for detecting the vehicles.

The main goals of this project are:

* Build a Histogram of Oriented GradientHOG) as feature extraction to be used as training data for a supervised learning system.
* Implement a classification system that finds which windows are an image of a car.
* Implement a sliding window search to detect vehicles.
* Implement a system to consolidate multiple detections into one window
* Implement a pipeline with all previous steps.
* Run the pipeline through different videos, tracking the different vehicles.
* Run a second pipeline sequentially where we delimit the lanes.

[car_nocar]: ./output_images/car-nocar.png
[car_nocar_hog]: ./output_images/car-nocar-hog.png
[classification_example]: ./output_images/classification-example.png
[window_car]: ./output_images/window-car.png
[window_heat]: ./output_images/window-heat.png
[video1]: ./results/project_video_vehicles.mp4

All the code is in the Jupyter notebook, and I will be specifying which methods are doing each piece of the work.

First we can visualize the training data we have (cars and no cars).

![alt text][car_nocar]

For training purposes, we could increase the training data by applying some data augmentation; initially I wanted to have the pipeline working before thinknig on adding data augmentation to improve classification on not available data.


## Build a Histogram of Oriented Gradients (HOG) as feature extraction to be used as training data for a supersvised learning system.

I implemented different versions of the HOG in this project, for different user cases. The main function for implementing hog features for an image is hog_features, that returns the hog features for the image given, using default number of orientations, pixels per cell and cell per block. Increasing the number of orientations seems to improve a lot the training results, even when increasing pixels per cell.

Below there is the original and the hog image for a car and a non-car.

![alt text][car_nocar_hog]

Building the HOG is just one way of getting features for the classifiers. The other two features recommended are a color histogram and a spatial reduction of the image. This step (and higher number of feature engineering) is fairly normal for traditional ML problems; in previous projects we have skipped this piece since the deep learning systems were creating the filters themselves and the first layers were creating the different features to be used by the classification. For this reason, when later I try to use a NN with these features, and since I am feeding the features already engineered, I feed directly to a FNN, instead of a CNN or RNN first layers.

The method features concatenates the three features for all the images provided; the appended single_image in the functions mean that the computation is applied to a single image.


## Implement a classification system that finds which windows are an image of a car

After computing the features for all car and non-car images, we need to implement a classifier that uses these features and that we can use to detect cars. I tested different classifiers using different color spaces (with multiple or single HOG features). Classification results improved specially with higher orientations.

With SVM, and using 24 orientations, 8 pixels per cell and 2 cells per block were:

* Test accuracyi (bin + hist + luv): 100.0000
* Test accuracy RGB: 99.92%
* Test accuracy HSV: 99.99%
* Test accuracy LUV: 100.00%
* Test accuracy HLS: 99.99%
* Test accuracy YUV: 100.00%
* Test accuracy YCrCb: 100.00%
* Test accuracy Random Forest LUV: 99.85%
* Test accuracy AdaBoost LUV: 98.71%
* Test accuracy AdaBoost YUV: 98.45%

And I implemented a small NN for testing using keras (function nn_model). It's a FNN with 4 fully connected hidden layers (512, 128, 32 and 16 neurons): 

* Test accuracy NN: 100%


## Implement a sliding window search to detect vehicles.

The implementation of the sliding window search follows a similar approach than the one stated on the lectures. In the sliding_windows I created all possible combinations of windows from a start and end point, with specific sizes and overlap. Later on, I call this function with windows of different sizes, populating a list with all possible_windows.

Finally, we can do a call to search windows to compute, for each window, if it belongs to an image of a car on not.

Below there is an example of a window with the label of the car (classified as car with a score of 99.64%).

![alt text][classification_example]


## Implement a system to consolidate multiple detections into one window

After this, we can use a heatmap to remove some false positives, and keep only the window intersections with enough windows. This is implemented with the functions add_heat and apply_threshold. Below there is an image with the image with all windows detected, the final window image and the heatmap image.


![alt text][window_car]

![alt text][window_heat]

## Run the pipeline through different videos, tracking the different vehicles.

Finally, I implemented the pipeline that will be used to compute the windows. For this pipeline, I am following the steps:

* Find the windows that are classified as cars in an image.
* Build the heatmaps for these windows.
* Apply threshold to the heatmaps.
* Compute the labels for the position of the cars.
* Draw the bounding boxes on the original image.

![alt text][video1]

The final videos after applying the pipeline are in the results folder. In the videos there are a few false positives on some areas, and in some cases, loses track of the car. There are different reasons for this to happen. One (more trivial solution) is to compute these false positives, get a few images from these and retrain the models with the new data. This is a manual process and it takes a fair amount of time (computing the features and retraining the models requires some work and plenty of computational time). I did this, adding a couple of thousand images from missclassifications.

A second method to improve tracking is to use previous frames to compute the position in the new frame, instead of just relying on one frame (since the final result can become really noisy, and we are keeping too many false negatives in the video).

Another improvement that I didn't apply in the pipeline is to compute once the HOG and read the windows from this precomputed HOG instead of cropping the image and then computing the HOG from it.

One of the most frustrating parts of the project was to ensure that the data format was consistent; in some cases the images were stored with an extra alpha channel (depending on the library used) and in some cases, the image was stored as a uint in the range [0, 255], in some others as a [0, 1] float.
