# Writeup

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
[image3]: ./output_images/slide_window.png
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4


---
## Writeup / README


You're reading it!

## Histogram of Oriented Gradients (HOG)

HOG features is the most important features in this Machine Learning project, it can provide a test accuracy of 0.99. While using other features but HOG features can only barely reach the test accuracy of 0.93

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


### 3. How the HOG parameters is settled.

I tried various combinations of parameters and final choice is a trade between the accuaracy and speed. I choose 9 `orientations`, 8 `pix_per_cell` and 2 `cell_per_block`.
This combination can provide enough accuracy and the computing speed is not too slow. While smaller parameters can result in smaller accuracy and larger parameters can make the training slow.

### 4.How the classifier is trainde using selected features.

I first extract the training data from the given database. I used all the pictures "that is a vehicel" but use only part of the non-vehicles pictures to form the training set, which is all the data from `./non-vehicles/Extras/*.png` and 1000 of the pictures from the 
`./non-vehicles/GTI/*.png`. This is because I want to make the total numbers of vehicles & not-vehicles are generally the same and the training data set can contain enough catalogs.

Then I set shuffle the data and split it into a training set and a test set by a protion of 0.2.
This is containde in the code block 5:

```python
    data_cars = glob.glob('./vehicles/*/*.png')
    data_notcars = glob.glob('./non-vehicles/Extras/*.png')
#     data_notcars_argu = glob.glob('./non-vehicles/GTI/*.png')
#     data_notcars_argu = shuffle(data_notcars_argu, n_samples=1000)
#     data_notcars.extend(data_notcars_argu)
    
    # set shuffle parameter to Ture to randomly shuffle the training data(although the default value of shuffle is True already...)
    data_cars_train, data_cars_test = train_test_split(data_cars, test_size=0.2, shuffle=True)
    data_notcars_train, data_notcars_test = train_test_split(data_notcars, test_size=0.2, shuffle=True)
    
```
Then I use the `clf = GridSearchCV(svr, parameters)` function to find a good parameters combination to fit the model(using 300 of pictures). The paremeters found is `kernal: rbf` and 'c: 5'.
Finally I trained the model and got a test accuracy of 0.9964

```
31.55 Seconds to search and train SVC
best combo {'C': 5, 'kernel': 'rbf'}
Test accuracy of SVC is 0.969
```
```
107 Seconds to train SVC
Test accuracy of SVC is 0.9964
```

## Sliding Window Search

### 1.How a sliding window search is implemented.  How did you decide what scales to search and how much to overlap windows?

I searched the lower part of the picture which is the part that vehicles can appear, which has the yscale from 400 to 700.
I first search the part using a 64x64 square window to search the ares.

![alt text][image3]

### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

## Video Implementation

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

