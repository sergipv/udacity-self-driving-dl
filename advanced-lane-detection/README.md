# Advanced Lane Finding

The goal of this project is to be able to find and display the lines from different videos.

The main steps to solve this problem, after camera callibration, are:

* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify the binary image into a birds-eye view.
* Create a threshold binary image with line information.
* Fit a polynomial to the binary image for each line.
* Determine what is the curvature of the lane, and what is the position of the vehicle with respect ot the center.
* Warp back into the original image, drawing the lane boundaries, and the numerical estimations of the lane curvature and vehicle position.


## Apply a distorition correction to raw images

We can compute the distortion of the camera with OpenCV, essentially using the findChessboardCorners to find the different corners of reference images taken with the same camera of different prespectives of a chessboard-like drawing. After computing these values, I store them into a pickle file to be downloaded later.

An example from a distorted (top) and undistorted (bottom) image are displayed below:

[image1]: ./results/camera_distorted.png "Distorted image (left) and undistorted (right) "
![alt text][image1] 

This code is in the method plot_undistorted of the notebook.

## Apply a perspective transform to rectify the binary image into a birds-eye view

The perspective transform correspond to first identifying which area of the original image we want to warp, and the destination area. To implement this transformation first we need to know which area we want to "zoom in"; that area corresponds to the road ahead of us. In a real scenario we would probably have to do some type of variable detection (as we can see in the harder challenge, the are we *should* crop changes with time), but in this specific proble we are mapping from and to known coordinates. The original and warped images are shown below:

[image2]: ./results/frame63.png "Unwarped image"
![alt text][image2] 

[image3]: ./results/frame63_warped.png "Warped image"
![alt text][image3] 

This code is in the Perspective Transform section of the notebook (method unwarp).

## Create a threshold binary image with line information.

For this project, I tested different options to create a binary image where we highlight the road lanes. After some testing, the color channels that seemed to contain more information were the L channel (from LUV), B channel (from LAB), S channel (from HSV) and a sobel x transform. After playing with some of the thresholds (highly dependent on a constant lighting for the camera), I removed the sobel x transform from the final threshold, since its signal was already added by other channels and it was adding some noise in the final binary image.

The different binary-thresholded images from the channels, and the combination of all partial binary images, are shown below:


[image4]: ./results/binaries.png "Binary images"
![alt text][image4] 

One post-processing step I realized after combining the multiple binary images was to apply a morphological close operation. This operation produces a dilation followed by an erosion. While this might strengthen noise on the binary images, it also strengthen the lines on images where they are not so easily detected. Initially I thought of performing an open operation, which would have removed most of the noise, but it ended up affecting the lines when the image was more difficult to process.

This (creating the binary image) is the most important during line detection. In some sense, it involves a lot of manual work, which is not ideal. Also, the channels used (and specially thresholds!) are not robust across different illuminations. I would have liked to spend more time enhancing original image to make easier (and more robust detection). Another technique that I would probably use to create a more robust system is Hough transfrom (which requires thinnes lines, so I would probably need a different binary selection that gets the edges from the binary lanes.

One strategy that I followed here was to also improve my detection in the areas where the estimation was not great in the videos. I store all the frames of the videos and checked why the estimation was not being great on some areas.

This code is in the Create Threshold section of the notebook (method combine).

## Fit a polynomial to the binary image for each line

I implemented 2 methods, each using the same windows-based approach. In the first method, I follow the same steps than specified in Udacity videos (window mode). First find a the start of both lines using a histogram, and then use windows to find the next position of the lines. By computing the "white points" in the different windows and doing a regression of these points, we can approximate the lines found to a polynomial (order 2) that we can then display.

In the second approach, the steps are similar, but instead of using all "white points" for the windows to estimate the lines I use a statistic per window to do this estimation (the mean). This way, we only have one point per window. And, in the case we don't see any new pixel in the current window. If no "white pixel" are in the current windows, we compute the increment between the two previous windows and assume we are having the same increment. This creates a more robust estimation of the lines.

Another problem during the estimation was the case there was some noise and the polynomial that we fit is way out; to improve this problem in both approaches I implemented a low pass filter basically keeping the last N polynomial estimations, with N=10, and using the mean of these values. Another strategy (only in the second approach) is to ignore the polynomial estimation for which the right and left lines cross before y=0; in this case I reuse the previous estimation.

In the image below the line estimation is shown, with the different window position, and the strategy to keep the same increments for non-detected lines, as long as some line was detected before.

[image5]: ./results/poly_dots.png "Estimation of the lines"
![alt text][image5] 


This code is in the methods sliding_sindow (sliding_window_pipeline and alternative_sliding_window).

## Determine what is the curvature of the lane and what is the position of the vehicle with respect to the center.

For this step I used the approximated polynomial (see curverad method in the code) to determine the curvature of the estimated line. This curvature changes depending on the video and time (if the line is straighter, the curvature increases, when the line gets more curve, the radius of curvature decreases to ~1Km.

To compute the distance to the center  of the road, I use the assumption that the camera is situated right on the center of the vehicle and that the meters per pixel at the bottom of the image is 3.7/700 m/pixel. With these approximations, we can compute the distance to the center by finding (x_leftline - x_rightline) / 2.

This code is in get_curverad and pipeline methods.

## Warp back into the original image, drawing the lane boundaries, and the numerical estimations of the lane curvature and vehicle position.

I was able to have pretty good results for two of the three videos. The harder_challenge video requires (or I think should require) more preprocessing and I didn't have time to add more steps into the pipeline. There are two main areas that I think we would need to focus:

* Rough detection of the road: using static positions for the road does not work anymore.
* Detection of cars and other object that might block the view on the image.
* Creation of the binary image (would need some preprocessing). In a deep learning sense, if we could create a nice training data set, we could use a GAN to generate binary images with the lines lighted up. We would need a fair amount of data, and the discriminant would compare the generated binary with a known binary, while the generator will create the binary that defines the lines. One adventage of this approach is that we could offuscate, change the illumination and modify all the input data to increase our training dataset.

See a final frame from the challenge video where this information is displayed:

[image6]: ./results/pipeline.png "Estimation of the lines in the pipeline"
![alt text][image6] 

This code is in the pipeline method of the notebook.
