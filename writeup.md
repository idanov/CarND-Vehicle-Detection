**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog-features-vehicle]: ./output_images/hog-features-vehicle.png
[hog-features-non-vehicle]: ./output_images/hog-features-non-vehicle.png
[window-sizes]: ./output_images/window-sizes.png
[test1]: ./output_images/test1.png
[test2]: ./output_images/test2.png
[test3]: ./output_images/test3.png
[test4]: ./output_images/test4.png
[test5]: ./output_images/test5.png
[test6]: ./output_images/test6.png
[frame1]: ./output_images/frame1.png
[frame2]: ./output_images/frame2.png
[frame3]: ./output_images/frame3.png
[frame4]: ./output_images/frame4.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

This is a writeup for addressing all rubric points. I have followed the template shown
[here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I have implemented a function called `img_hog_features()` for extracting the HOG features of one or more channels a given image.
The code for this function could be found in `image_helpers.py`, lines 53-66. As it could be seen from the code
this function makes use of another function called `get_hog_features()` where the actual feature extraction happens.

Extracting the features for all training images happens in the class `CarDetector`, defined in `car_detection.py`, lines 10-165.
This class has a private method, called `self.__extract_features()` which extracts all the features for a list of images, including
spatial features, color histogram and HOG features. Extracting the features for a single image is done by `single_img_features()`
defined in `image_helpers.py`, lines 90-114, and that is where `img_hog_features()` is used.

Here is how the HOG features of a `vehicle` and `non-vehicle` images look like:

![Vehicle][hog-features-vehicle]

![Non-vehicle][hog-features-non-vehicle]

#### 2. Explain how you settled on your final choice of HOG parameters.

My choice of HOG parameters was done after trying out different color spaces
and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`)
for training the ML model. After a few iterations of manually tweaking the parameters,
I settled on the best options I could find: `YCrCb` color space and HOG parameters of
`orientations=16`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For training the model, I have used a feature vector constructed from concatenating
`(16, 16)` spatial features, color histogram with `32` bins and flattened
HOG features from the previous section. All of these were done for each channel of
the image, converted in `YCrCb` color space. The total length of the feature
vector ended up being `2592`.

I used a 85/15 percent training/testing data split of the provided images.
For the `non-vehicle`, the choice of testing data was random. The `vehicle` images
however were split more carefully because the data from the GTI directories
was extracted from sequential video frames and a random split would compromise our test data.
Therefore all GTI directories were considered separately and the last 15% from each directory
was dedicated to test data. The test images from the KITTI were selected randomly.

The resulting numbers for train/test data sizes ended up being fairly balanced
between `vehicle` and `non-vehicle` images:
```
Non-car images (train/test): 7622 / 1346
Car images (train/test): 7471 / 1321
Total train/test split: 14942 / 2642
```

All the code for that can be found in `data_loading.py`, where only two functions
are defined: `load_vehicle_lists()` and `split_data()`.

The model used for the classification was a linear SVM model and it
achieved test accuracy of `0.9909` with the default parameters:
```
Feature vector length: 2592
8.23 seconds to train SVC...
Test Accuracy of SVC =  0.9909
My SVC predicts:  [1 0 1 0 0 1 1 0 0 0]
For these 10 labels:  (1, 0, 1, 0, 0, 1, 1, 0, 0, 0)
0.00555 seconds to predict 10 labels with SVC
```

All the code for the model training can be found in the method `self.train_model()`
of the class `CarDetector` in `car_detection.py`, lines 45-76.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

After manually inspecting the test images, I concluded that all cars appear
within pixels `400` and `656` from the top of the image. Many window sizes
could contribute positively, but I settled on the following sizes (first element in the tuples):
```
win_sizes = [(48, 2, 560),
             (64, 2, 576),
             (96, 1, 608),
             (128, 1, 624)]
```
The window steps were defined as the second element of the tuple in terms of
HOG cells (each cell was chosen to be `16px` as mentioned earlier).

The window steps have to be defined in terms of HOG cells, because the HOG features
were extracted only once for the whole region and then used many times by different
windows in order to speed up the image processing.

The third element in the tuple is the coordinate of the bottom of the region
of interest (ROI). The choice of different subregions reflect the fact that cars appear smaller
only around the top of the ROI, where larger objects can be detected
closer to the bottom of the image.

The code for performing the sliding window search could be found in the class `CarDetector`
in file `car_detection.py`, methods `self.detect_cars()`, `self.__search_windows()` and `self.__slide_window()`.

Here is an image showing the different window sizes:

![Window sizes][window-sizes]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

For improving the performance of the classifier, I relied on choosing good parameters
for the feature extraction. The resulting parameters were mentioned in the section above.
Here are some example images:

![Test1][test1]

![Test2][test2]

![Test3][test3]

![Test4][test4]

![Test5][test5]

![Test6][test6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.
From the positive detections I created a heatmap and then thresholded that map agressively (`thresh=3`)
to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.
I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

I kept a history of these bounding boxes for 5 frames and used that history
to go through the earlier mentioned procedure again, this time using all potential
vehicle bounding boxes for the current frame and the past 5 frames.

The finally detected bounding boxes of the blobs are displayed on the frame
as detected vehicles. The code for all that could be found in `video_gen.py`,
in functions `windows_to_cars()` and `process_image()`, lines 28-57.

Here are the first four frames of a video and their corresponding detected boxes, detected bounding boxes,
thresholded heatmaps (including history), labels (including history) and resulting bounding boxes (including history):

![Frame1][frame1]

![Frame2][frame2]

![Frame3][frame3]

![Frame4][frame4]

Note that the first frame with resulting bounding boxes is frame 4. That is due to
applied history thresholding for bounding boxes - they need to appear in at least
3 frames before being drawn as resulting bounding boxes.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

During the development I faced issues with classifying many false positives,
even though the validation accuracy was `>0.99` on the test set. I used only
the training data given for the assignment and maybe using the Udacity training
data could have helped to improve the generalization to video frames.

Another drawback of the current approach is that scanning for cars happen
with only a small number of preselected window sizes, which results in not so smooth
detections. A possible improvement could be trying to apply a Deep Learning approach
with a CNN going through the whole image, which should yield a more scale and
translation invariant approach.

Another problem is that the speed of frame processing is very slow due to calculating the HOG features.
Ideally we would like to process frames in real time, so more work for optimizing the processing
could be done.
