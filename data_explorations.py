# dummy
import numpy as np
import cv2

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from skimage.feature import hog

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from scipy.ndimage.measurements import label

import pickle
import glob
from random import shuffle
import os
import time
import math

from tqdm import tqdm
from moviepy.editor import VideoFileClip

vehicles_far = glob.glob('./vehicles/GTI_Far/image*.png')
vehicles_left = glob.glob('./vehicles/GTI_Left/image*.png')
vehicles_middle = glob.glob('./vehicles/GTI_MiddleClose/image*.png')
vehicles_right = glob.glob('./vehicles/GTI_Right/image*.png')
vehicles = vehicles_far + vehicles_left + vehicles_middle + vehicles_right
shuffle(vehicles)

non_vehicles_extras = glob.glob('./non-vehicles/Extras/extra*.png')
non_vehicles_gti = glob.glob('./non-vehicles/GTI/image*.png')
non_vehicles = non_vehicles_extras + non_vehicles_gti
shuffle(non_vehicles)


# Define a function to get the info of the images
def get_data_info(cars, not_cars):
	data_dict = {}
	data_dict['num_cars'] = len(cars)
	data_dict['num_not_cars'] = len(not_cars)

	img = mpimg.imread(cars[0])
	data_dict['img_shape'] = img.shape
	data_dict['d_type'] = img.dtype

	print('Number of Vehicle Pics: {0}'.format(data_dict['num_cars']))
	print('Number of Non-vehicle Pics: {0}'.format(data_dict['num_not_cars']))
	print('Img Shape: {0}'.format(data_dict['img_shape']))
	print('Img Type: {0}'.format(data_dict['d_type']))

	return data_dict



# Define a function to show some samples from the images
def show_sample_images(cars, not_cars):
	fig = plt.figure(figsize=(20, 10))
	n = 0
	while n < 10:
		# select random image
		car_rand_num = np.random.randint(0, len(cars))
		non_car_rand_num = np.random.randint(0, len(not_cars))

		car_img = mpimg.imread(cars[car_rand_num])
		non_car_img = mpimg.imread(not_cars[non_car_rand_num])

		fig.add_subplot(1, 10, n+1)
		plt.imshow(car_img)
		plt.xticks(())
		plt.yticks(())
		plt.title('Vehicle')

		fig.add_subplot(2, 10, n+1)
		plt.imshow(non_car_img)
		plt.xticks(())
		plt.yticks(())
		plt.title('Non-Vehicle')

		n += 1

	plt.savefig('output_images/sample_car_non_car_images.png', bbox_inches="tight")


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

# Define a function to show histogram of a random image
def create_sample_histogram_img(cars):
	choice = np.random.randint(0, len(cars))
	img = cv2.imread(cars[choice])
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	red_hist, green_hist, blue_hist, bin_centers, hist_features = color_hist(img=img)

	fig = plt.figure(figsize = (10, 3))
	matplotlib.rc('xtick', labelsize=20)
	matplotlib.rc('ytick', labelsize=20)

	fig = plt.figure(figsize=(10,3));
	matplotlib.rc('xtick', labelsize=20)
	matplotlib.rc('ytick', labelsize=20)

	plt.subplot(1, 4, 1)
	plt.imshow(img)
	plt.title('Original Image:\n', fontsize=20);

	plt.subplot(1, 4, 2)
	plt.bar(bin_centers, red_hist[0], width=3)
	plt.xlim(0, 256)
	plt.title('Red:\n', fontsize=20);

	plt.subplot(1, 4, 3)
	plt.bar(bin_centers, green_hist[0], width=3)
	plt.xlim(0, 256)
	plt.title('Green:\n', fontsize=20);

	plt.subplot(1, 4, 4)
	plt.bar(bin_centers, blue_hist[0], width=3)
	plt.xlim(0, 256)
	plt.title('Blue:\n', fontsize=20);

	plt.subplots_adjust(left=0.5, right=2, top=1, bottom=0.)
	plt.savefig('output_images/sample_histogram_image.png', bbox_inches="tight")

	return img

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

def plot_org_and_3d(img, pixels, colors_rgb,
				axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):

	fig = plt.figure(figsize=(20, 10))
	ax_1 = fig.add_subplot(1, 2, 1)
	fig.tight_layout()
	matplotlib.rc('xtick', labelsize=30)
	matplotlib.rc('ytick', labelsize=30)
	ax_1.imshow(img)
	ax_1.set_title('Original Image:\n', fontsize=30)

	ax_2 = fig.add_subplot(1, 2, 2, projection='3d')
	ax_2.text2D(0.15, 0.99, "Color Distribution:\n",
					transform=ax_2.transAxes, fontsize=30)

	# Set axis limits
	ax_2.set_xlim(*axis_limits[0])
	ax_2.set_ylim(*axis_limits[1])
	ax_2.set_zlim(*axis_limits[2])

	# Set axis labels and sizes
	ax_2.tick_params(axis='both', which='major', labelsize=30, pad=8)
	ax_2.set_xlabel(axis_labels[0], fontsize=30, labelpad=20)
	ax_2.set_ylabel(axis_labels[1], fontsize=30, labelpad=20)
	ax_2.set_zlabel(axis_labels[2], fontsize=30, labelpad=20)

	# Plot pixel values with colors given in colors_rgb
	ax_2.scatter(
		pixels[:, :, 0].ravel(),
		pixels[:, :, 1].ravel(),
		pixels[:, :, 2].ravel(),
		c=colors_rgb.reshape((-1, 3)), edgecolors='none')

	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
	plt.savefig('output_images/sample_color_space_image.png', bbox_inches="tight")

def create_sample_color_space_img(cars, rand = 0, img=None):
	if rand != 0:
		choice = np.random.randint(0, len(cars))
		img = cv2.imread(cars[choice])
		# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Select a small fraction of pixels to plot by subsampling it
	scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
	img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

	# Convert subsampled image to desired color space(s)
	img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
	img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
	img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

	# Plot and show
	# plot3d(img_small_RGB, img_small_rgb)
	# plt.show()

	# plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
	# plt.show()

	plot_org_and_3d(img, img_small_RGB, img_small_rgb)

	return img

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

def create_sample_binning_image(cars, rand = 0, img=None):
	if rand != 0:
		choice = np.random.randint(0, len(cars))
		img = cv2.imread(cars[choice])

	feature_image, features = bin_spatial(img, size=(32, 32))

	fig = plt.figure(figsize=(20, 10))
	ax_1 = fig.add_subplot(1, 3, 1)
	fig.tight_layout()
	matplotlib.rc('xtick', labelsize=20)
	matplotlib.rc('ytick', labelsize=20)
	ax_1.imshow(img)
	ax_1.set_title('Original Image:\n', fontsize=20)

	ax_2 = fig.add_subplot(1, 3, 2)
	ax_2.plot(features)
	ax_2.set_title('Spatial Binning:\n', fontsize=20)

	ax_3 = fig.add_subplot(1, 3, 3)
	small_img = cv2.resize(img, (32, 32))
	ax_3.imshow(small_img)
	ax_3.set_title('32 x 32:\n', fontsize=20)

	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
	plt.savefig('output_images/sample_spatial_binning_image.png', bbox_inches="tight")

	return img


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

def create_sample_gradient_image(cars, rand=0, img=None):
	if rand != 0:
		choice = np.random.randint(0, len(cars))
		img = cv2.imread(cars[choice])

	binary_output = gradient_features(img)

	fig = plt.figure(figsize=(12, 5))
	ax_1 = fig.add_subplot(1, 2, 1)
	fig.tight_layout()
	matplotlib.rc('xtick', labelsize=20)
	matplotlib.rc('ytick', labelsize=20)
	ax_1.imshow(img)
	ax_1.set_title('Original Image:\n', fontsize=20)

	ax_2 = fig.add_subplot(1, 2, 2)
	ax_2.imshow(binary_output, cmap='gray')
	ax_2.set_title('Gradient Image:\n', fontsize=20)

	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
	plt.savefig('output_images/sample_gradient_image.png', bbox_inches='tight')

	return img

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

def create_sample_hog_image(caars, rand=0, img=None):
	if rand != 0:
		choice = np.random.randint(0, len(cars))
		img = cv2.imread(cars[choice])

	hog_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	features, hog_image = get_hog_features(hog_img, visualise=True)

	fig = plt.figure(figsize=(12, 5))

	fig.add_subplot(1, 2, 1)
	matplotlib.rc('xtick', labelsize=20)
	matplotlib.rc('ytick', labelsize=20)
	plt.imshow(img)
	plt.title('Original Image:\n', fontsize=20)

	fig.add_subplot(1, 2, 2)
	plt.imshow(hog_image, cmap='hot')
	plt.title('HOG Image:\n', fontsize=20)
	plt.savefig('output_images/sample_hog_image.png', bbox_inches='tight')

	return img


def extract_features_pipeline(images,
					color_space='RGB',
					histogram_feature=True,
					spatial_feature=True,
					hog_feature=True,
					spatial_size=(32, 32),
					hist_bins=32,
					hist_range=(0, 256),
					orientations=9,
					pixels_per_cell=(8,8),
					cells_per_block=(2,2),
					hog_channel=0):


	features = []

	for file in tqdm(images):
		img = mpimg.imread(file)

		file_features = []

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
		else:
			feature_image = np.copy(img)

		# spatial
		if spatial_feature == True:
			spatial_feature_image, spatial_features = bin_spatial(img, size=spatial_size)
			file_features.append(spatial_features)

		# histogram
		if histogram_feature == True:
			rhist, ghist, bhist, bin_centers, hist_features = color_hist(img, nbins=hist_bins)
			file_features.append(hist_features)

		# hog
		if hog_feature == True:
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,channel],
													orientations=orientations,
													pixels_per_cell=pixels_per_cell,
													cells_per_block=cells_per_block,
													visualise=False,
													feature_vector=True))
				hog_features = np.ravel(hog_features)
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel],
											orientations=orientations,
											pixels_per_cell=pixels_per_cell,
											cells_per_block=cells_per_block,
											visualise=False,
											feature_vector=True)

			file_features.append(hog_features)

		features.append(np.concatenate(file_features))

	return features

def extract_features_pipeline_for_single_image(img,
					color_space='RGB',
					histogram_feature=True,
					spatial_feature=True,
					hog_feature=True,
					spatial_size=(32, 32),
					hist_bins=32,
					hist_range=(0, 256),
					orientations=9,
					pixels_per_cell=(8,8),
					cells_per_block=(2,2),
					hog_channel=0):

	file_features = []

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
	else:
		feature_image = np.copy(img)

	# spatial
	if spatial_feature == True:
		spatial_feature_image, spatial_features = bin_spatial(img, size=spatial_size)
		file_features.append(spatial_features)

	# histogram
	if histogram_feature == True:
		rhist, ghist, bhist, bin_centers, hist_features = color_hist(img, nbins=hist_bins)
		file_features.append(hist_features)

	# hog
	if hog_feature == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend(get_hog_features(feature_image[:,:,channel],
												orientations=orientations,
												pixels_per_cell=pixels_per_cell,
												cells_per_block=cells_per_block,
												visualise=False,
												feature_vector=True))
			# hog_features = np.ravel(hog_features)
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel],
										orientations=orientations,
										pixels_per_cell=pixels_per_cell,
										cells_per_block=cells_per_block,
										visualise=False,
										feature_vector=True)

		file_features.append(hog_features)

	return np.concatenate(file_features)



def extract_cars_and_non_cars_features(cars, not_cars):
	car_features = extract_features_pipeline(cars)
	non_car_features = extract_features_pipeline(not_cars)
	return car_features, non_car_features


def create_sample_scaled_features_image(cars, not_cars):
	choice = np.random.randint(0, len(cars))

	car_features, non_car_features = extract_cars_and_non_cars_features(cars, not_cars)

	if len(car_features) > 0:
		X = np.vstack((car_features, non_car_features)).astype(np.float64)
		X_scaler = StandardScaler().fit(X)
		scaled_X = X_scaler.transform(X)

		fig = plt.figure(figsize=(20, 5))
		matplotlib.rc('xtick', labelsize=20)
		matplotlib.rc('ytick', labelsize=20)

		plt.subplot(1, 3, 1)
		plt.imshow(mpimg.imread(cars[choice]))
		plt.title('Original Image:\n', fontsize=20)

		plt.subplot(1, 3, 2)
		plt.plot(X[choice])
		plt.title('Raw Features:\n', fontsize=20)

		plt.subplot(1, 3, 3)
		plt.plot(scaled_X[choice])
		plt.title('Normalized Features:\n', fontsize=20)

		plt.savefig('output_images/sample_scaled_features_image.png')

	else:
		print('Empty Feature Vectors')

def data_exploration(cars, not_cars):
	get_data_info(cars, not_cars)
	show_sample_images(cars, not_cars)
	img = create_sample_histogram_img(cars)
	img = create_sample_color_space_img(cars, 0, img)
	img = create_sample_binning_image(cars, 0, img)
	img = create_sample_gradient_image(cars, 0, img)
	img = create_sample_hog_image(cars, 0, img)

	create_sample_scaled_features_image(cars, not_cars)


def data_preprocess(cars, not_cars):
	car_features, non_car_features = extract_cars_and_non_cars_features(cars, not_cars)

	X = np.vstack((car_features, non_car_features)).astype(np.float64)
	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)

	y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

	print('scaled_X: {0}'.format(scaled_X.shape))
	print('y: {0}'.format(y.shape))

	# split the data
	rand_state = np.random.randint(0, 10)
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,test_size=0.2,random_state=rand_state)
	print('After splitting,')
	print('X_train: {0}'.format(X_train.shape))

	return X_train, X_test, y_train, y_test


def svm_classifier(X_train, X_test, y_train, y_test):
	print('Training the SVM Classifier')
	svc = LinearSVC(C=0.01)
	t = time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print('Seconds to train: {0}'.format(round(t2-t, 2)))
	print('Test Accuracy: {0:0.4f}%'.format(svc.score(X_test, y_test)*100))
	print('  Predictions:', svc.predict(X_test[0:20]))
	print('       Labels:', y_test[0:20])

	return svc



def draw_boxes(img, bboxes, color=(0,0,255), thickness=6):
	img_copy = np.copy(img)
	for bbox in bboxes:
		cv2.rectangle(img_copy, bbox[0], bbox[1],
						color, thickness)
	return img_copy



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
        y_start_stop[0] = int(img.shape[0]*0.45)
    if y_start_stop[1] == None:
        y_start_stop[1] = int(img.shape[0]*0.9)
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
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

def create_sample_windows_image():
	test_img = mpimg.imread('test_images/test5.jpg')
	windows = slide_window(test_img)
	window_img = draw_boxes(test_img, windows)
	plt.imshow(window_img)
	matplotlib.rc('xtick', labelsize=15)
	matplotlib.rc('ytick', labelsize=15)
	plt.title('Slideing Windows:\n', fontsize=15)
	plt.savefig('output_images/sample_windows_image_test5.png')

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
		features = extract_features_pipeline_for_single_image(test_img,
								color_space=color_space,
								histogram_feature=hist_feat,
								spatial_feature=spatial_feat,
								hog_feature=hog_feat,
								spatial_size=spatial_size,
								hist_bins=hist_bins,
								orientations=orient,
								pixels_per_cell=pix_per_cell,
								cells_per_block=cell_per_block,
								hog_channel=hog_channel)
		#5) Scale extracted features to be fed to classifier
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		#6) Predict using your classifier
		prediction = clf.predict(test_features)
		#7) If positive (prediction == 1) then save the window
		if prediction == 1:
			on_windows.append(window)
	#8) Return windows for positive detections
	return on_windows


def whole_pipeline(cars, not_cars):
	color_space = 'LUV'
	hist_feat = True
	spatial_feat = True
	hog_feat = True
	spatial_size = (16, 16)
	hist_bins = 16
	orientations = 9
	pixels_per_cell = (16, 16)
	cells_per_block = (4, 4)
	hog_channel = 'ALL'
	y_start_stop = [None, None]

	car_features  = extract_features_pipeline(cars,
						color_space=color_space,
						histogram_feature=hist_feat,
						spatial_feature=spatial_feat,
						hog_feature=hog_feat,
						spatial_size=spatial_size,
						hist_bins=hist_bins,
						hist_range=(0, 256),
						orientations=orientations,
						pixels_per_cell=pixels_per_cell,
						cells_per_block=cells_per_block,
						hog_channel=hog_channel)

	non_car_features = extract_features_pipeline(not_cars,
						color_space=color_space,
						histogram_feature=hist_feat,
						spatial_feature=spatial_feat,
						hog_feature=hog_feat,
						spatial_size=spatial_size,
						hist_bins=hist_bins,
						hist_range=(0, 256),
						orientations=orientations,
						pixels_per_cell=pixels_per_cell,
						cells_per_block=cells_per_block,
						hog_channel=hog_channel)

	X = np.vstack((car_features, non_car_features)).astype(np.float64)
	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)

	y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

	print('scaled_X: {0}'.format(scaled_X.shape))
	print('y: {0}'.format(y.shape))

	# split the data
	rand_state = np.random.randint(0, 10)
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,test_size=0.2,random_state=rand_state)
	print('Using:', orientations, 'orientations', pixels_per_cell,
		'pixels per cell and', cells_per_block, 'cells per block')
	print('Feature vector length:', len(X_train[0]))

	svc_classifier = svm_classifier(X_train, X_test, y_train, y_test)

	img = mpimg.imread('test_images/test4.jpg')
	draw_img = np.copy(img)

	# Training Data extracted from .png images (scaled 0 to 1 by mpimg)
	# Search image is a .jpg (scaled 0 to 255)
	img = img.astype(np.float32)/255

	windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
							xy_window=(64, 64), xy_overlap=(0.85, 0.85))

	hot_windows = search_windows(img, windows, svc_classifier, X_scaler, color_space=color_space,
								spatial_size=spatial_size, hist_bins=hist_bins,
								orient=orientations, pix_per_cell=pixels_per_cell,
								cell_per_block=cells_per_block,
								hog_channel=hog_channel, spatial_feat=spatial_feat,
								hist_feat=hist_feat, hog_feat=hog_feat)

	window_img = draw_boxes(draw_img, hot_windows)


	f = plt.figure(figsize=(20, 5))
	matplotlib.rc('xtick', labelsize=30)
	matplotlib.rc('ytick', labelsize=30)

	plt.subplot(1, 2, 1)
	plt.imshow(img)
	plt.title('Original Image:\n', fontsize=30);
	plt.subplot(1, 2, 2)
	plt.imshow(window_img)
	plt.title('Bounding Boxes:\n', fontsize=30);

	plt.savefig('output_images/all_window_detections.png')

	bbox_pickle = {}
	all_bboxes = hot_windows
	bbox_pickle["bboxes"] = all_bboxes
	pickle.dump(bbox_pickle, open("output_images/bbox_pickle.p", "wb"));

	return window_img, hot_windows

def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap

def apply_threshold(heatmap, threshold=4):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

def create_sample_heat_map_image(window_img=None, all_bboxes=None):
	if (all_bboxes == None):
		bbdict = pickle.load( open( "output_images/bbox_pickle.p", "rb" ))
		all_bboxes = bbox_pickle["bboxes"]

	img = mpimg.imread('test_images/test4.jpg')
	heatmap = np.zeros_like(img[:,:,0].astype(np.float))

	heatmap = add_heat(heatmap, all_bboxes)
	heatmap = apply_threshold(heatmap)
	heatmap = np.clip(heatmap-2, 0, 255)

	labels = label(heatmap)

	fig = plt.figure(figsize=(20, 5))
	matplotlib.rc('xtick', labelsize=20)
	matplotlib.rc('ytick', labelsize=20)

	plt.subplot(1, 2, 1)
	plt.imshow(window_img)
	plt.title('Bounding Boxes:\n', fontsize=20)

	plt.subplot(1, 2, 2)
	plt.imshow(heatmap, cmap='hot')
	plt.title('Heatmap:\n', fontsize=20)
	plt.savefig('output_images/sample_heatmap_image.png')

	return labels

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


def create_sample_labeled_image(labels):
	img = mpimg.imread('test_images/test4.jpg')
	draw_img = np.copy(img)
	draw_img = draw_labeled_bboxes(draw_img, labels)

	fig = plt.figure(figsize=(20, 5))
	matplotlib.rc('xtick', labelsize=20)
	matplotlib.rc('ytick', labelsize=20)
	plt.imshow(draw_img)
	plt.title('Vehicle Bounding Boxes:\n', fontsize=20)

	plt.savefig('output_images/sample_labeled_image.png')


data_exploration(vehicles, non_vehicles)
# X_train, X_test, y_train, y_test = data_preprocess(vehicles, non_vehicles)
# svc_classifier = svm_classifier(X_train, X_test, y_train, y_test)
# create_sample_windows_image()






# window_img, bbox_list = whole_pipeline(vehicles, non_vehicles)
# labels = create_sample_heat_map_image(window_img, bbox_list)
# create_sample_labeled_image(labels)
