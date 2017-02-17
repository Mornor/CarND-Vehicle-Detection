"""
Contains all of the tools used in ./src/process.py
Author: Celien Nanson <cesliens@gmail.com>
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import csv

# Useful constants 
DATASET_PATH = './dataset/object-detection-crowdai/'

def load_images():
	"""
	Load images into memory and separate between vehicule and non-vehicule
	"""

	car_images_path = []
	not_car_images_path = []
	car_images = []
	not_car_images = []

	# First divide the original .csv between cars and not_cars path
	with open(DATASET_PATH + 'labels.csv', 'r') as f:
		reader = csv.reader(f)
		f.readline() # Skip first line
		for row in reader:
			if row[5] == 'Car':
				car_images_path.append(row)
			else: 
				not_car_images_path.append(row)

	# Extract individual vehicules and non-vehicules from the image
	for i in range(0, 1000):
		image = mpimg.imread(DATASET_PATH + car_images_path[i][4])
		xmin = int(car_images_path[i][0])
		xmax = int(car_images_path[i][2])
		ymin = int(car_images_path[i][3])
		ymax = int(car_images_path[i][1])
		width = xmax - xmin
		height = ymin - ymax # inverted axes 
		car = image[ymax:ymax+height, xmin:xmin+width]
		car = cv2.resize(car, (64, 64))
		car_images.append(car)

	for i in range(0, 1000):
		image = mpimg.imread(DATASET_PATH + not_car_images_path[i][4])
		xmin = int(not_car_images_path[i][0])
		xmax = int(not_car_images_path[i][2])
		ymin = int(not_car_images_path[i][3])
		ymax = int(not_car_images_path[i][1])
		width = xmax - xmin
		height = ymin - ymax # inverted axes 
		not_car = image[ymax:ymax+height, xmin:xmin+width]
		not_car = cv2.resize(not_car, (64, 64))
		not_car_images.append(not_car)

	# Transform it into np array
	car_images = np.asarray(car_images)
	not_car_images = np.asarray(not_car_images)

	#plot_image(car_images[2])

	#print(car_images.shape)
	#print(not_car_images.shape)

	return car_images, not_car_images

# Compute binned color features  
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel() 
	# Return the feature vector
	return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	if vis == True:
		features, hog_image = hog(img, orientations=orient, 
								pixels_per_cell=(pix_per_cell, pix_per_cell),
								cells_per_block=(cell_per_block, cell_per_block), 
								transform_sqrt=True, 
								visualise=vis, feature_vector=feature_vec)
		return features, hog_image

	else:      
		features = hog(img, orientations=orient, 
					pixels_per_cell=(pix_per_cell, pix_per_cell),
					cells_per_block=(cell_per_block, cell_per_block), 
					transform_sqrt=True, 
					visualise=vis, feature_vector=feature_vec)
		return features

def extract_features(images, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
	 # Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for image in images:
		file_features = []
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

		if spatial_feat:
			spatial_features = bin_spatial(feature_image, size=spatial_size)
			file_features.append(spatial_features)

		if hist_feat == True:
			hist_features = color_hist(feature_image, nbins=hist_bins)
			file_features.append(hist_features)

		if hog_feat == True:
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)        
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)

			file_features.append(hog_features)
		features.append(np.concatenate(file_features))

	# Return list of feature vectors
	return features

def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, 
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						spatial_feat=True, hist_feat=True, hog_feat=True):  
						  
	img_features = []
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
	
	if spatial_feat:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		img_features.append(spatial_features)

	if hist_feat:
		hist_features = color_hist(feature_image, nbins=hist_bins)
		img_features.append(hist_features)

	if hog_feat:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))      
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		img_features.append(hog_features)

	return np.concatenate(img_features)

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.75, 0.75)):
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

def normalize(car_features, not_car_features): 
	"""
	:return the normalized feature vector
	"""
	X = np.vstack((car_features, not_car_features)).astype(np.float64)                        
	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)
	return scaled_X, X_scaler


def plot_image(image, gray=False):
	if(gray):
		plt.imshow(image, cmap='gray')
	else: 
		plt.imshow(image)
	plt.show()

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), 
					orient=9, pix_per_cell=8, cell_per_block=2, 
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

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	# Make a copy of the image
	imcopy = np.copy(img)
	# Iterate through the bounding boxes
	for bbox in bboxes:
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	return imcopy
