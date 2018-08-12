# Writeup Template


---

**Vehicle Detection Project**

The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a SVM classifier
* A color transform and append binned color features is also applied, which is the Y channel of the YUV color spaces, as well as histograms of color, to your HOG feature vector. 
* Using svm Machine Learning to train a model to predict the picture is a vehicle or not.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_orig.png
[image2]: ./output_images/hog_show.png
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4


---
## Writeup / README


You're reading it!

## Histogram of Oriented Gradients (HOG)

HOG features is the most important features in this Machine Learning project, it can provide a test accurancy of 0.99. While using other features but HOG features can only barely reach the test accuracy of 0.93

### 1.Extract the HOG features and histogram & binned color features from the training images.

The code for this step is contained in the second code cell of the IPython notebook.

```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # all default arguments
    return_list = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), \
                               cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, \
                               visualise = vis, feature_vector=feature_vec, block_norm='L2-Hys')
    
    if vis == True:
        hog_img = return_list[1]
        return return_list[0], hog_img
    else:
        return return_list
```
Before the HOG features is extracted, I convert the image into grayscale using `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` and then extract the HOG features from the grayscale picture. The color channel convertion is not included in the function above, however, it is done before the picture is feed into this function.

Below is the visualization of the HOG features of an example

![alt text][image1]
![alt text][image2]

### 2. Extract histogram and binned color features from the picture

Then the histogram and binned color features from the picture, using the function in code block 2.

```python
def bin_spatial(img, size):
    # using luminance color channel from yuv color spaces 
#     img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_luma = img[:,:,0]
    color_features = cv2.resize(img_luma, (size,size)).ravel()
    return color_features

# function to compute color histogram features
def color_hist(img, nbins, bin_range=(0,256)):
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bin_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bin_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bin_range)
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    return hist_features
```

Finally a function is usend in code block 2 to extract and combine thoes features together.

```python
# define a function to extract featrues from a single feature
def single_features(img, nbins, size, orient, pix_per_cell, cell_per_block):
    # create a list to append feature vectors to
    features = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_forhog = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # histogram features
    hist_features = color_hist(img, nbins)
    # color space featuresorient, pix_per_cell, cell_per_block, block_n
    color_features = bin_spatial(img, size)
    # hog features of every channel in the image
    hog_features = []
    
    hog_features.append(get_hog_features(img_forhog, orient, pix_per_cell, cell_per_block, vis=False))
    hog_features = np.ravel(hog_features)
    features = np.hstack((hist_features, color_features, hog_features))
    
    return features

# funciton to extract features from an image
def extract_features(imgs, nbins, size, orient, pix_per_cell, cell_per_block):
    # create a list to append feature vectors to
    features = []
    # iterate through the list of images
    for file in imgs:
        file_features = []
        img = mpimg.imread(file)
        file_features = single_features(img, 32, 32, 9, 8, 2)
        features.append(file_features)
    
    return features
```

You can see that I changed the RGB picture to YUV color spaces and used the Y channel as a feature, for the Y channel is insulated from the colors and can represent the vehicle shape.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

