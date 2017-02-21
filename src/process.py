"""
Process the video stream fram by frame to detect vehicules
Use ./src/utils.py
Author: Celien Nanson <cesliens@gmail.com>
"""

import cv2
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import numpy as np
import utils

# Parameters
color_space = 'YUV'		# Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  				# HOG orientations
pix_per_cell = 8 			# HOG pixels per cell
cell_per_block = 2 			# HOG cells per block
hog_channel = "ALL" 		# Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) 	# Spatial binning dimensions
hist_bins = 32  			# Number of histogram bins
spatial_feat = True 		# Spatial features on or off
hist_feat = True 			# Histogram features on or off
hog_feat = True 			# HOG features on or off
y_start_stop = [380, 550]	# Min and max in y to search in slide_window()
x_start_stop = [800, 1280]	# Min and max in x to search in slide_window()

class Detector:
	def __init__(self, svc, X_scaler):
		self.svc = svc 
		self.X_scaler = X_scaler

	def process_frame(self, image):
		# Test the result on one single image
		draw_image = np.copy(image)

		windows = utils.slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, xy_window=(96, 96), xy_overlap=(0.75, 0.75))

		hot_windows = utils.search_windows(image, windows, self.svc, self.X_scaler, color_space=color_space, 
								spatial_size=spatial_size, hist_bins=hist_bins, 
								orient=orient, pix_per_cell=pix_per_cell, 
								cell_per_block=cell_per_block, 
								hog_channel=hog_channel, spatial_feat=spatial_feat, 
								hist_feat=hist_feat, hog_feat=hog_feat)                       

		window_img = utils.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

		# Find the place were is the most overlapping boxes by drawing a Heatmap
		heat = np.zeros_like(window_img[:,:,0]).astype(np.float)
		heat = utils.add_heat(heat, hot_windows)
		heat = utils.apply_threshold(heat, 1)
		heatmap = np.clip(heat, 0, 255)
		labels = label(heatmap)
		draw_img = utils.draw_labeled_bboxes(image, labels)

		return draw_img


def prepare_data():
	# Load the images as RGB, 64x64
	print('Loading data into memory...')
	car_images, not_car_images = utils.load_images(udacity=False)
	print('Data loaded')

	# Extract useful feaures from images 
	car_images_features = utils.extract_features(car_images, color_space=color_space, 
							spatial_size=spatial_size, hist_bins=hist_bins, 
							orient=orient, pix_per_cell=pix_per_cell, 
							cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_feat=spatial_feat, 
							hist_feat=hist_feat, hog_feat=hog_feat)

	not_car_images_features = utils.extract_features(not_car_images, color_space=color_space, 
							spatial_size=spatial_size, hist_bins=hist_bins, 
							orient=orient, pix_per_cell=pix_per_cell, 
							cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_feat=spatial_feat, 
							hist_feat=hist_feat, hog_feat=hog_feat)

	# Normalize the features
	X, X_scaler, scaled_X = utils.normalize(car_images_features, not_car_images_features)

	# Define the labels vector 
	y = np.hstack((np.ones(len(car_images_features)), np.zeros(len(not_car_images_features))))

	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

	# Train the SVC classifier
	#print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
	#print('Feature vector length:', len(X_train[0]))
	svc = LinearSVC()
	svc.fit(X_train, y_train)
	
	return svc, X_scaler
	#print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

def process(image, svc, X_scaler):
	# Test the result on one single image
	image = mpimg.imread(image)
	draw_image = np.copy(image)

	windows = utils.slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, xy_window=(96, 96), xy_overlap=(0.75, 0.75))

	hot_windows = utils.search_windows(image, windows, svc, X_scaler, color_space=color_space, 
							spatial_size=spatial_size, hist_bins=hist_bins, 
							orient=orient, pix_per_cell=pix_per_cell, 
							cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_feat=spatial_feat, 
							hist_feat=hist_feat, hog_feat=hog_feat)                       

	window_img = utils.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

	# Find the place were is the most overlapping boxes by drawing a Heatmap
	heat = np.zeros_like(window_img[:,:,0]).astype(np.float)
	heat = utils.add_heat(heat, hot_windows)
	heat = utils.apply_threshold(heat, 1)
	heatmap = np.clip(heat, 0, 255)
	labels = label(heatmap)
	draw_img = utils.draw_labeled_bboxes(image, labels)

	return draw_img

svc, X_scaler = prepare_data()
print('Training SVC...')
detector = Detector(svc, X_scaler)
print('Done training SVC')

'''
image_1 = mpimg.imread('./test_images/test1.jpg')
image_2 = mpimg.imread('./test_images/test2.jpg')
image_3 = mpimg.imread('./test_images/test3.jpg')
image_4 = mpimg.imread('./test_images/test4.jpg')
image_5 = mpimg.imread('./test_images/test5.jpg')
image_6 = mpimg.imread('./test_images/test6.jpg')
result_1 = detector.process_frame(image_1)
result_2 = detector.process_frame(image_2)
result_3 = detector.process_frame(image_3)
result_4 = detector.process_frame(image_4)
result_5 = detector.process_frame(image_5)
result_6 = detector.process_frame(image_6)
plt.imshow(result_1)
plt.imshow(result_2)
plt.imshow(result_3)
plt.imshow(result_4)
plt.imshow(result_5)
plt.imshow(result_6)
plt.show()
'''

output = 'output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(detector.process_frame) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)