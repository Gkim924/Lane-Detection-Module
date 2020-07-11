import time
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
# from sensor_msgs.msg import Image
# from std_msgs.msg import Header
# from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import Float32
from skimage import morphology

def gradient_thresh(img, thresh_min=25, thresh_max=100):
	"""
	Apply sobel edge detection on input image in x, y direction
	"""
	#1. Convert the image to gray scale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#2. Gaussian blur the image
	gray = cv2.GaussianBlur(gray, (5,5), 0)
	#3. Use cv2.Sobel() to find derievatives for both X and Y Axis
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
	#4. Use cv2.addWeighted() to combine the results
	gray = cv2.addWeighted(np.absolute(sobelx), 0.5, np.absolute(sobely), 0.5, 0)
	#5. Convert each pixel to unint8, then apply threshold to get binary image	
	#gray = cv2.convertScaleAbs(gray)
	gray = np.uint8(gray)
	num, binary_output = cv2.threshold(gray, thresh_min, thresh_max, cv2.THRESH_BINARY)
	# binary_output = gray

	return binary_output


def color_thresh(img, thresh=(100, 255)):
	"""
	Convert RGB to HSL and threshold to binary image using S channel
	"""
	#1. Convert the image from RGB to HSL
	hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	#2. Apply threshold on S channel to get binary image
	binary_output = cv2.inRange(hsl, (0, 0, thresh[0]), (255, 255, thresh[1]))

	return binary_output

def combinedBinaryImage(img):
	"""
	Get combined binary image from color filter and sobel filter
	"""
	#1. Apply sobel filter and color filter on input image
	SobelOutput = gradient_thresh(img)
	ColorOutput = color_thresh(img)
	#2. Combine the outputs
	# SobelOutput = cv2.addWeighted(SobelOutput, 0.5, ColorOutput, 0.5, 0)
	## Here you can use as many methods as you want.

	## TODO

	####

	binaryImage = np.zeros_like(SobelOutput)
	binaryImage[(ColorOutput!=0)|(SobelOutput!=0)] = 1
	# Remove noise from binary image
	binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)

	binaryImage = binaryImage.astype(np.uint8)*255

	return binaryImage


def perspective_transform(img, verbose=False):
	"""
	Get bird's eye view from input image
	"""
	#1. Visually determine 4 source points and 4 destination points
	#2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
	#3. Generate warped image in bird view using cv2.warpPerspective()

	## TODO
	#src_pts = np.float32([[300,250],[842,250],[1031,370],[210,370]])
	#dest_pts = np.float32([[0,0],[300,0],[300,300],[0,300]])
	src_pts = np.float32([[480,250],[680,250],[809,370],[292,370]])
	dest_pts = np.float32([[292,0],[809,0],[809,370],[292,370]])

	M = cv2.getPerspectiveTransform(src_pts,dest_pts)

	warped_img = cv2.warpPerspective(img,M,(1242,375))

	####

	#return warped_img, M, Minv
	return warped_img

# Lane Fitting
def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[100:midpoint]) + 100
	rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		##TO DO
		win_y_low = binary_warped.shape[0] - (window + 1)* window_height
		win_y_high = binary_warped.shape[0] - window * window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		####
		# Draw the windows on the visualization image using cv2.rectangle()
		##TO DO
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2)
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2)
		####
		# Identify the nonzero pixels in x and y within the window
		##TO DO
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		####
		# Append these indices to the lists
		##TO DO
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		####
		# If you found > minpix pixels, recenter next window on their mean position
		##TO DO
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
		####
		pass

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each using np.polyfit()
	##TODO
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	####

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	cv2.imshow("Fit", out_img)
	cv2.waitKey(0)

	return ret


if __name__ == '__main__':
	# Import the image
	img = cv2.imread(r"test.png")
	cv2.imshow("Orig Img", img)
	cv2.waitKey(0)
	
	#Test for the gradient thresh
	thresh_img = gradient_thresh(img)
	cv2.imshow("Gradient Thresh Img", thresh_img)
	cv2.waitKey(0)

	# Test for the color thresh
	thresh_img2 = color_thresh(img)
	cv2.imshow("Color Thresh Img", thresh_img2)
	cv2.waitKey(0)

	# Test for the combined image
	img2 = combinedBinaryImage(img)
	cv2.imshow("Combined Img", img2)
	cv2.waitKey(0)

	# Test for perspective tform
	img_perspective = perspective_transform(img2)
	cv2.imshow("Perspective Tform Img", img_perspective)
	cv2.waitKey(0)

	line_fit(img_perspective)

	













	
