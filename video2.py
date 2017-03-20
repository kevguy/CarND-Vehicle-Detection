
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
# from lesson_functions import *
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
# import imageio
# imageio.plugins.ffmpeg.download()
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

from collections import deque
from scipy.ndimage.measurements import label

from skimage.feature import hog



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



























# Load the classifier and parameters
data_file = 'ClassifierData.p'
with open(data_file, mode='rb') as f:
	data = pickle.load(f)

svc = data['svc']
X_scaler = data['X_scaler']
color_space = data['color_space']
spatial_size = data['spatial_size']
hist_bins = data['hist_bins']
orient = data['orient']
pix_per_cell = data['pix_per_cell']
cell_per_block = data ['cell_per_block']
hog_channel = data['hog_channel']
spatial_feat = data ['spatial_feat']
hist_feat = data['hist_feat']
hog_feat = data['hog_feat']

def search_all_scales(img):
	hot_windows = []
	all_windows = []

	X_start_stop =[[None,None],[None,None]]
	Y_start_stop =[[390,470],[390,500]]
	XY_window = [(64,64),(110,110)]
	XY_overlap=[(0.75, 0.75),(0.75, 0.75)]

	X_start_stop =[[None,None],[None,None],[None,None],[None,None]]
	w0,w1,w2,w3 = 240,180,120,70
	o0,o1,o2,o3 = 0.75,0.75,0.75,0.75
	XY_window = [(w0,w0),(w1,w1),(w2,w2),(w3,w3)]
	XY_overlap = [(o0,o0),(o1,o1),(o2,o2),(o3,o3)]
	yi0,yi1,yi2,yi3 = 380,380,395,405
	Y_start_stop =[[yi0,yi0+w0/2],[yi1,yi1+w1/2],[yi2,yi2+w2/2],[yi3,yi3+w3/2]]

	for i in range(len(Y_start_stop)):
		windows = slide_window(img,
								x_start_stop=X_start_stop[i],
								y_start_stop=Y_start_stop[i],
								xy_window=XY_window[i],
								xy_overlap=XY_overlap[i])
		all_windows += [windows]
		hot_windows += search_windows(img, windows, svc, X_scaler,
										color_space=color_space,
										spatial_size=spatial_size,
										hist_bins=hist_bins,
										orient=orient,
										pix_per_cell=pix_per_cell,
										cell_per_block=cell_per_block,
										hog_channel=hog_channel,
										spatial_feat=spatial_feat,
										hist_feat=hist_feat,
										hog_feat=hog_feat)
	return hot_windows, all_windows


def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	if bbox_list:
		for box in bbox_list:
			# Add += 1 for all pixels inside each bbox
			# Assuming each "box" takes the form ((x1, y1), (x2, y2))
			heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap

def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	# Return the image
	return img


# Define a class to receive the characteristics of bounding box detections
class BoundingBoxes:
    def __init__(self,n=10):
        # length of queue to store data
        self.n = n
        # hot windows of the last n images
        self.recent_boxes = deque([],maxlen=n)
        # current boxes
        self.current_boxes = None
        self.allboxes = []

    def add_boxes(self):
        self.recent_boxes.appendleft(self.current_boxes)

    def pop_data(self):
        if self.n_buffered>0:
            self.recent_boxes.pop()
    def set_current_boxes(self,boxes):
        self.current_boxes = boxes

    def get_all_boxes(self):
        allboxes = []
        for boxes in self.recent_boxes:
            allboxes += boxes
        if len(allboxes)==0:
            self.allboxes = None
        else:
            self.allboxes = allboxes

    def update(self,boxes):
        self.set_current_boxes(boxes)
        self.add_boxes()
        self.get_all_boxes()


boxes = BoundingBoxes(n=20)

def video_pipeline(img):
	draw_image = np.copy(img)
	img = img.astype(np.float32)/255
	hot_windows, _ = search_all_scales(img)
	boxes.update(hot_windows)
	heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
	heatmap = add_heat(heatmap, boxes.allboxes)
	heatmap = apply_threshold(heatmap, 17)
	labels = label(heatmap)

	window_image = draw_labeled_bboxes(draw_image, labels)
	#window_image = draw_boxes(draw_image, hot_windows, color=(0,0,255), thick=6)
	return window_image

out_dir='./output_images/'

inpfile='project_video.mp4'
outfile=out_dir+'processed_'+inpfile
clip = VideoFileClip(inpfile)
out_clip = clip.fl_image(video_pipeline)
out_clip.write_videofile(outfile, audio=False)

# inpfile='test_video.mp4'
# outfile=out_dir+'processed_'+inpfile
# clip = VideoFileClip(inpfile)
# out_clip = clip.fl_image(video_pipeline)
# out_clip.write_videofile(outfile, audio=False)


print('fuck')
images = sorted(glob.glob('./test_images/*.jpg'))
print(images)
count = 0
for file in images:
	print('you suck')
	image = mpimg.imread(file)
	image = image.astype(np.float32)/255
	draw_image = np.copy(image)

	t=time.time()
	hot_windows,all_windows = search_all_scales(image)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to search windows ...')
	#print(np.array(all_windows).shape)
	window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 1), thick=4)



	allwindows_img = draw_image
	for ind,win_list in enumerate(all_windows):
		if ind==0: color= (0,0,1)
		if ind==1: color= (0,1,0)
		if ind==2: color= (1,0,0)
		if ind==3: color= (1,1,1)

		allwindows_img = draw_boxes(allwindows_img, all_windows[ind], color=color, thick=6)



	plt.figure()
	# Plot the result
	f, (ax1,ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()

	ax1.imshow(window_img)
	ax1.set_title('Detected windows', fontsize=40)

	ax2.imshow(allwindows_img)
	ax2.set_title('All windows', fontsize=40)
	# plt.show()
	print('you suck')
	plt.savefig('./output_images/new_sliding_windows' + str(count) + '.png')
	count += 1


images = sorted(glob.glob('./test_images/*.jpg'))
boxes = BoundingBoxes(n=6)
count = 0
for file in images:
    image = mpimg.imread(file)
    image = image.astype(np.float32)/255
    draw_image = np.copy(image)

    t=time.time()
    hot_windows,all_windows = search_all_scales(image)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to search windows ...')

    boxes.update(hot_windows)

    #print(np.array(all_windows).shape)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 1), thick=4)

    allwindows_img = draw_image
    for ind,win_list in enumerate(all_windows):
        if ind==0: color= (0,0,1)
        if ind==1: color= (0,1,0)
        if ind==2: color= (1,0,0)
        if ind==3: color= (1,1,1)

        allwindows_img = draw_boxes(allwindows_img, all_windows[ind], color=color, thick=6)


    # Read in the last image shown above
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    heatmap = add_heat(heatmap, boxes.allboxes)
    heatmap  = apply_threshold(heatmap,3)

    plt.figure()
    # Plot the result
    f, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(window_img)
    ax1.set_title('Detected windows', fontsize=40)

    ax2.imshow(allwindows_img)
    ax2.set_title('All windows', fontsize=40)

    ax3.imshow(heatmap)
    ax3.set_title('Heatmap', fontsize=40)
    plt.savefig('./output_images/new_heat_map' + str(count) + '.png')
    count += 1
