import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.externals import joblib


# lession_functions

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
						vis=False, feature_vec=True):
	# Call with two outputs if vis==True
	if vis == True:
		features, hog_image = hog(img, orientations=orient, 
									pixels_per_cell=(pix_per_cell, pix_per_cell),
									cells_per_block=(cell_per_block, cell_per_block), 
									transform_sqrt=True, 
									visualise=vis, feature_vector=feature_vec)
		return features, hog_image
	# Otherwise call with one output
	else:      
		features = hog(img, orientations=orient, 
						pixels_per_cell=(pix_per_cell, pix_per_cell),
						cells_per_block=(cell_per_block, cell_per_block), 
						transform_sqrt=True, 
						visualise=vis, feature_vector=feature_vec)
		return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel() 
	# Return the feature vector
	return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
						hist_bins=32, orient=9, 
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						spatial_feat=True, hist_feat=True, hog_feat=True):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for file in imgs:
		file_features = []
		# Read in each one by one
		image = mpimg.imread(file)
		# apply color conversion if other than 'RGB'
		if color_space != 'RGB':
			if color_space == 'HSV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif color_space == 'LUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif color_space == 'HLS':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif color_space == 'YUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif color_space == 'YCrCb':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
		else: feature_image = np.copy(image)      

		if spatial_feat == True:
			spatial_features = bin_spatial(feature_image, size=spatial_size)
			file_features.append(spatial_features)
		if hist_feat == True:
			# Apply color_hist()
			hist_features = color_hist(feature_image, nbins=hist_bins)
			file_features.append(hist_features)
		if hog_feat == True:
		# Call get_hog_features() with vis=False, feature_vec=True
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,channel], 
										orient, pix_per_cell, cell_per_block, 
										vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)        
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
							pix_per_cell, cell_per_block, vis=False, feature_vec=True)
			# Append the new feature vector to the features list
			file_features.append(hog_features)
		features.append(np.concatenate(file_features))
	# Return list of feature vectors
	return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
					xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
	# Compute the span of the region to be searched    
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	# Compute the number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
	# Compute the number of windows in x/y
	nx_windows = np.int(xspan/nx_pix_per_step) - 1
	ny_windows = np.int(yspan/ny_pix_per_step) - 1
	# Initialize a list to append window positions to
	window_list = []
	# Loop through finding x and y window positions
	# Note: you could vectorize this step, but in practice
	# you'll be considering windows one by one with your
	# classifier, so looping makes sense
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
            
			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
	# Return the list of windows
	return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
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
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows











































vehicles_far = glob.glob('./vehicles/GTI_Far/image*.png')
vehicles_left = glob.glob('./vehicles/GTI_Left/image*.png')
vehicles_middle = glob.glob('./vehicles/GTI_MiddleClose/image*.png')
vehicles_right = glob.glob('./vehicles/GTI_Right/image*.png')
vehicles_ext = glob.glob('./vehicles/KITTI_extracted/*.png')

non_vehicles_extras = glob.glob('./non-vehicles/Extras/extra*.png')
non_vehicles_gti = glob.glob('./non-vehicles/GTI/image*.png')
non_vehicles = non_vehicles_extras + non_vehicles_gti


# split 70% training 20% validation 10% test set
frac1 = 0.7
l0,l1,l2,l3,l4,l5=len(vehicles_far),len(vehicles_left),len(vehicles_middle),len(vehicles_right),len(vehicles_ext),len(non_vehicles)
L1 = (frac1*np.array([l0,l1,l2,l3,l4,l5])).astype('int')
frac2 = 0.9
l0,l1,l2,l3,l4,l5=len(vehicles_far),len(vehicles_left),len(vehicles_middle),len(vehicles_right),len(vehicles_ext),len(non_vehicles)
L2 = (frac2*np.array([l0,l1,l2,l3,l4,l5])).astype('int')

cars_train = vehicles_far[:L1[0]] + vehicles_left[:L1[1]] + vehicles_middle[:L1[2]] + vehicles_right[:L1[3]] + vehicles_ext[:L1[4]]
notcars_train = non_vehicles[:L1[5]]

cars_val = vehicles_far[L1[0]:L2[0]] + vehicles_left[L1[1]:L2[1]] + vehicles_middle[L1[2]:L2[2]] + vehicles_right[L1[3]:L2[3]] + vehicles_ext[L1[4]:L2[4]]
notcars_val = non_vehicles[L1[5]:L2[5]]

cars_test = vehicles_far[L2[0]:] + vehicles_left[L2[1]:] + vehicles_middle[L2[2]:] + vehicles_right[L2[3]:] + vehicles_ext[L2[4]:]
notcars_test = non_vehicles[L2[5]:]


print('Number of samples in cars training set: ', len(cars_train))
print('Number of samples in notcars training set: ', len(notcars_train))

print('Number of samples in cars validation set: ', len(cars_val))
print('Number of samples in notcars validation set: ', len(notcars_val))

print('Number of samples in cars test set: ',len(cars_test))
print('Number of samples in notcars test set: ',len(notcars_test))

def get_features(files, color_space='RGB', spatial_size=(32, 32),
					hist_bins=32, orient=9, 
					pix_per_cell=8, cell_per_block=2, hog_channel=0,
					spatial_feat=True, hist_feat=True, hog_feat=True):
	features = []
	for file in files:
        
		img = mpimg.imread(file)
		img_features = single_img_features(img, color_space=color_space, spatial_size=spatial_size,
						hist_bins=hist_bins, orient=orient,
						pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
						spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        
		features.append(img_features)
	return features

color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (16, 16)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_feat = True
hist_feat = True
hog_feat = True

t=time.time()
cars_train_feat = get_features(cars_train,color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
cars_val_feat = get_features(cars_val,color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
cars_test_feat = get_features(cars_test,color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

notcars_train_feat = get_features(notcars_train,color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
notcars_val_feat = get_features(notcars_val,color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
notcars_test_feat = get_features(notcars_test,color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG,spatial and color features...')

# Create an array stack of feature vectors
X = np.vstack((cars_train_feat,cars_val_feat,cars_test_feat,
				notcars_train_feat,notcars_val_feat,notcars_test_feat)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

cars_ntrain=len(cars_train_feat)
cars_nval=len(cars_val_feat)
cars_ntest=len(cars_test_feat)
ncars_ntrain=len(notcars_train_feat)
ncars_nval=len(notcars_val_feat)
ncars_ntest=len(notcars_test_feat)

i1 = cars_ntrain
i2 = i1 + cars_nval
i3 = i2 + cars_ntest
i4 = i3 + ncars_ntrain
i5 = i4 + ncars_nval

cars_train_feat,cars_val_feat,cars_test_feat = scaled_X[:i1],scaled_X[i1:i2],scaled_X[i2:i3]
notcars_train_feat,notcars_val_feat,notcars_test_feat = scaled_X[i3:i4],scaled_X[i4:i5],scaled_X[i5:]

y_train = np.hstack((np.ones(cars_ntrain), np.zeros(ncars_ntrain)))
y_val = np.hstack((np.ones(cars_nval), np.zeros(ncars_nval)))
y_test = np.hstack((np.ones(cars_ntest), np.zeros(ncars_ntest)))

X_train = np.vstack((scaled_X[:i1],scaled_X[i3:i4]))
X_val = np.vstack((scaled_X[i1:i2],scaled_X[i4:i5]))
X_test = np.vstack((scaled_X[i2:i3],scaled_X[i5:]))

X_train,y_train = shuffle(X_train,y_train,random_state=42)
X_val,y_val = shuffle(X_val,y_val,random_state=42)
X_test,y_test = shuffle(X_test,y_test,random_state=42)


print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()


# use of the rbf kernel improves the accuracy by about another percent, 
# but increases the prediction time up to 1.7s(!) for 100 labels. Too slow.
#svc = svm.SVC(kernel='rbf')

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Validation Accuracy of SVC = ', round(svc.score(X_val, y_val), 4))
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 100
print('My SVC predicts: ', svc.predict(X_val[0:n_predict]))
print('For these',n_predict, 'labels: ', y_val[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


# pickle_file = 'ProcessedData.p'
# print('Saving data to pickle file...')
# try:
# 	with open(pickle_file, 'wb') as pfile:
# 		pickle.dump(
# 			{
# 				'X_train': X_train,
# 				'X_val': X_val,
# 				'X_test': X_test,
# 				'y_train': y_train,
# 				'y_val': y_val,
# 				'y_test': y_test                
# 			},
# 			pfile, pickle.HIGHEST_PROTOCOL)
# except Exception as e:
# 	print('Unable to save data to', pickle_file, ':', e)
# 	raise
    
# print('Data cached in pickle file.')


pickle_file = 'ClassifierData.p'
print('Saving data to pickle file...')
try:
	with open(pickle_file, 'wb') as pfile:
		pickle.dump(
			{   'svc':svc, 
				'X_scaler': X_scaler,
				'color_space': color_space,
				'spatial_size': spatial_size,
				'hist_bins': hist_bins,
				'orient': orient,
				'pix_per_cell': pix_per_cell,
				'cell_per_block': cell_per_block,
				'hog_channel': hog_channel,
				'spatial_feat': spatial_feat,
				'hist_feat': hist_feat,
				'hog_feat':hog_feat
			},
			pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
	print('Unable to save data to', pickle_file, ':', e)
	raise

print('Data cached in pickle file.')
