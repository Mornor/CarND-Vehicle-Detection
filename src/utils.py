"""
Contains all of the tools used in ./src/process.py
Author: Celien Nanson <cesliens@gmail.com>
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
import csv

# Useful constants 
DATASET_PATH = './dataset/object-detection-crowdai/'

def load_images():
	"""
	Load images into memory and separate between vehicule and non-vehicule
	"""

	cars_images_path = []
	not_cars_images_path = []
	cars_images = []
	not_cars_images = []

	# First divide the original .csv between cars and not_cars path
	with open(DATASET_PATH + 'labels.csv', 'r') as f:
		reader = csv.reader(f)
		f.readline() # Skip first line
		for row in reader:
			if row[5] == 'Car':
				cars_images_path.append(row)
			else: 
				not_cars_images_path.append(row)

	# Extract individual vehicules and non-vehicules from the image
	for i in range(0, 1):
		image = mpimg.imread(DATASET_PATH + cars_images_path[i][4])
		xmin = int(cars_images_path[i][0])
		xmax = int(cars_images_path[i][1])
		ymin = int(cars_images_path[i][2])
		ymax = int(cars_images_path[i][3])
		height = ymax - ymin 
		width = xmax - xmin
		vehicule = image[ymax:ymin+height, xmax:xmin+width]
		plt.scatter(xmin, ymax)
		plt.scatter(xmax, ymin)

		plt.imshow(image)
		plt.show()
		#vehicule = extract_region_of_interest(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), vertices)

		#vehicule = extract_region_of_interest(image, vertices)
		#cars_images.append(vehicule)
		#not_cars_images.append(mpimg.imread(DATASET_PATH + not_cars_images_path[i][4]))

	# For each items, load the images into memory
	#cars_images = np.array([mpimg.imread(DATASET_PATH + file[4]) for file in cars_images_path]) 
	#not_cars_images = np.array([mpimg.imread(DATASET_PATH + file[4]) for file in not_cars_images_path]) 

	# Transform it into np array
	cars_images = np.asarray(cars_images)
	#not_cars_images = np.asarray(not_cars_images)
	#print(cars_images.shape)
	#plot_image(cars_images[0])

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	imcopy = np.copy(img)
	for bbox in bboxes:
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	return imcopy

def extract_region_of_interest(image, vertices):
	"""
	Extract a region of interest from the images defined by the box coordinates
	"""
	# Defining a blank mask to start with
	mask = np.zeros_like(image)

	# Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(image.shape) > 2:
		channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	# Filling pixels inside the polygon defined by "vertices" with the fill color
	cv2.fillPoly(mask, vertices, ignore_mask_color)

	# Feturning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(image, mask)
	
	return masked_image

def plot_image(image, gray=False):
	if(gray):
		plt.imshow(image, cmap='gray')
	else: 
		plt.imshow(image)
	plt.show()

'''
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	# Call with two outputs if vis==True
	if vis:
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
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
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
		else: 
			feature_image = np.copy(image)      

		if spatial_feat:
			spatial_features = bin_spatial(feature_image, size=spatial_size)
			file_features.append(spatial_features)
		
		if hist_feat:
			# Apply color_hist()
			hist_features = color_hist(feature_image, nbins=hist_bins)
			file_features.append(hist_features)
        
		if hog_feat:
		# Call get_hog_features() with vis=False, feature_vec=True
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)
			
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            
			# Append the new feature vector to the features list
			file_features.append(hog_features)
        
		features.append(np.concatenate(file_features))
    
	return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
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
'''
