def test_channel_hog(): 
	cars = glob.glob('./dataset/vehicles/*/*.png')
	notcar = glob.glob('./dataset/non-vehicles/*/*.png')

	image_original = imread(cars[5])
	image = utils.change_color_space(image_original, 'YCrCb')
	hog_images=[]
	for channel in range(image.shape[2]):
		_, hog_image = utils.get_hog_features(image[:,:,channel], vis=True)
		hog_images.append(hog_image)

	f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(8,3))
	f.tight_layout()
	ax1.imshow(image_original)
	ax1.set_title('Example Car Image')
	ax2.imshow(hog_images[0], cmap='gray')
	ax2.set_title('Y Channel Hog')
	ax3.imshow(hog_images[1], cmap='gray')
	ax3.set_title('Cr Channel Hog')
	ax4.imshow(hog_images[2], cmap='gray')
	ax4.set_title('Cb Channel Hog')
	plt.show()

def test_normalize(): 
	# Parameters
	color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9  # HOG orientations
	pix_per_cell = 8 # HOG pixels per cell
	cell_per_block = 2 # HOG cells per block
	hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
	spatial_size = (16, 16) # Spatial binning dimensions
	hist_bins = 16  # Number of histogram bins
	spatial_feat = False # Spatial features on or off
	hist_feat = False # Histogram features on or off
	hog_feat = True # HOG features on or off
	y_start_stop = [380, 680] # Min and max in y to search in slide_window()

	# Load the images as RGB, 64x64
	car_images, not_car_images = utils.load_images(udacity=False)

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

	if len(car_images_features) > 0:
		car_ind = np.random.randint(0, len(car_images))
		X, X_scaler, scaled_X = utils.normalize(car_images_features, not_car_images_features)
		fig = plt.figure(figsize=(12,4))
		plt.subplot(131)
		plt.imshow(car_images[car_ind])
		plt.title('Original Image')
		plt.subplot(132)
		plt.plot(X[car_ind])
		plt.title('Raw Features')
		plt.subplot(133)
		plt.plot(scaled_X[car_ind])
		plt.title('Normalized Features')
		fig.tight_layout()
		plt.show()
	else: 
		print('Your function only returns empty feature vectors...')