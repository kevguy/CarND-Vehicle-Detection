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

	feature_image, features = bin_spatial(img, size=(8, 8))

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
	small_img = cv2.resize(img, (8, 8))
	ax_3.imshow(small_img)
	ax_3.set_title('8 x 8:\n', fontsize=20)

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


def create_sample_scaled_features_image(cars, not_cars):
	choice = np.random.randint(0, len(cars))

	car_features = extract_features_pipeline(cars)
	non_car_features = extract_features_pipeline(not_cars)

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

def data_explorataon:
	get_data_info(vehicles, non_vehicles)
	show_sample_images(vehicles, non_vehicles)
	img = create_sample_histogram_img(vehicles)
	img = create_sample_color_space_img(vehicles, 0, img)
	img = create_sample_binning_image(vehicles, 0, img)
	img = create_sample_gradient_image(vehicles, 0, img)
	img = create_sample_hog_image(vehicles, 0, img)

	create_sample_scaled_features_image(vehicles, non_vehicles)

data_explorataon()





