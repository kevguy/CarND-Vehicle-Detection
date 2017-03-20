#Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[sample_histogram]: https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/sample_histogram_image.png
[sample_color_space]: https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/sample_color_space_image.png
[sample_spatial_binning]: https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/sample_spatial_binning_image.png
[sample_gradient]: https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/sample_gradient_image.png
[sample_hog]: https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/sample_hog_image.png
[heat_map0]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_heat_map0.png
[heat_map1]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_heat_map1.png
[heat_map2]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_heat_map2.png
[heat_map3]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_heat_map3.png
[heat_map4]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_heat_map4.png
[heat_map5]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_heat_map5.png
[sliding_window0]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_sliding_windows0.png
[sliding_window1]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_sliding_windows1.png
[sliding_window2]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_sliding_windows2.png
[sliding_window3]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_sliding_windows3.png
[sliding_window4]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_sliding_windows4.png
[sliding_window5]:https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/new_sliding_windows5.png
### Project 5 - Vehicle Detection and Tracking
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### 0: Files
 * **data_explorations.py** : for making images to be shown here
 * **data_process.py** : for the whole data preprocession part and training the SVC classifier
 * **video2.py** : for producing sliding windows and heatmaps and processing the video

### 1: Data Preprocession

#### 1.1: Feature Extraction
A couple of techniques were adopted for feature extractions which catered to better feature vectors, which are listed below:

**1.1.1: Histograms of Color**

The histograms of an image gives us a feature vector with concatenated features from RGB channels based on number of intensity bins and pixel intensity ranges.

```
# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features
```

![sample_histogram]

**1.1.2: Color Distribution**

Through analyzing the color distribution of an image, we can find cluster of colors which usually help us locate the position of an object.

```
def plot3d(pixels, colors_rgb,
		axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
	"""Plot pixels in 3D."""

	# Create figure and 3D axes
	fig = plt.figure(figsize=(8, 8))
	ax = Axes3D(fig)

	# Set axis limits
	ax.set_xlim(*axis_limits[0])
	ax.set_ylim(*axis_limits[1])
	ax.set_zlim(*axis_limits[2])

	# Set axis labels and sizes
	ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
	ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
	ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
	ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

	# Plot pixel values with colors given in colors_rgb
	ax.scatter(
		pixels[:, :, 0].ravel(),
		pixels[:, :, 1].ravel(),
		pixels[:, :, 2].ravel(),
		c=colors_rgb.reshape((-1, 3)), edgecolors='none')

	return ax  # return Axes3D object for further manipulation
```

![sample_color_space]

**1.1.3: Spatial Binning**

Spatial binning can lower the resolution of an image while still making sure an object is identifiable. This can potentially speed up training.

```
# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='RGB', size=(32, 32)):
	# Convert image to new color space (if specified)
	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else: feature_image = np.copy(img)
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(feature_image, size).ravel()
	# Return the feature vector
	return feature_image, features
```

![sample_spatial_binning]

**1.1.4: Gradient Magnitude**

The previous techniques were only manipulating and transforming color values and they only capture one aspect of an object's appearance. Gradients can give us a better presentation of the shape of the object, which is independent of colors.

```
def gradient_features(img, sobel_kernel=9, mag_threshold=(60, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	gradmag = np.sqrt(sobel_x**2 + sobel_y**2)
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_threshold[0]) &
					(gradmag <= mag_threshold[1])] = 1
	return binary_output
```

![sample_gradient]

**1.1.5: Histogram of Oriented Gradients (HOG)**

```
def get_hog_features(img,
				orientations=9,
				pixels_per_cell=(8,8),
				cells_per_block=(2,2),
				visualise=False,
				feature_vector=True):

	if (visualise == True):
		features, hog_image = hog(img,
						orientations=orientations,
						pixels_per_cell=pixels_per_cell,
						cells_per_block=cells_per_block,
						transform_sqrt=False,
						visualise=visualise,
						feature_vector=False)
		return features, hog_image
	else:
		features = hog(img,
						orientations=orientations,
						pixels_per_cell=pixels_per_cell,
						cells_per_block=cells_per_block,
						transform_sqrt=False,
						visualise=visualise,
						feature_vector=feature_vector)
		return features

```

![sample_hog]

#### 1.2 Training and Testing Set
Data is normalized, randomized and splitted into training set and testing as follows:

|                 | Cars    | Not cars | Total  |
|:-------------   |:-------:| --------:| ------:|
| Training set    | 6152    |     6277 | 12429  |
| Validation set  | 1759    |     1794 |  3553  |
| Testing set     | 881     |      897 |  1778  |

### 2. SVC Classifier
After the step of preprocessing, the data is fed into a SVC classifier to classify between 'vehicles' and 'non-vehicles'. On average, the classifier gives an accuracy of 98%.

### 3. Sliding Windows Search
An image is 'divided' into four partially overlapping zones with different sliding window sizes to account for different distances, which are 240, 180, 12, 70 pixels for each zone.

![sliding_window3]

![sliding_window4]

### 4. Heatmap
The boxes for every last 30 frames are saved, and then heatmaps are added to help eliminate false positives by applying a threshold of 17. Bounding boxes are then finally drawn to cover the area of each vehicle detected, which equivalently are the highlighted areas on the heatmaps.

![heat_map3]

![heat_map4]

## 5. Video
[output_images/processed_project-video.mp4](https://github.com/kevguy/CarND-Vehicle-Detection/raw/master/output_images/processed_project_video.mp4)



## 6. Discussion
I spent a lot of time tuning the parameters, like heatmaps' threshold, and the only way that I could finally make everything work is by adding and tuning how many frames I have to save for the boxes. To be frank, I was kinda surprised how much of a role SVC classifier and HOG play in this task.
