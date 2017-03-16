# dummy
import numpy as np
import cv2

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from skimage.feature import hog

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

get_data_info(vehicles, non_vehicles)


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


show_sample_images(vehicles, non_vehicles)