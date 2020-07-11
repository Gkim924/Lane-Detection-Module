import time
import math
import numpy as np
import cv2
# import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
# from sensor_msgs.msg import Image
# from std_msgs.msg import Header
# from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import Float32
from skimage import morphology

def gradient_thresh(img, thresh_min=64, thresh_max=255):
	"""
	Apply sobel edge detection on input image in x, y direction
	"""
	#1. Convert the image to gray scale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#2. Gaussian blur the image
	gray = cv2.GaussianBlur(gray, (5,5), 0)
	#3. Use cv2.Sobel() to find derievatives for both X and Y Axis
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)#, ksize=1)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)#, ksize=1)
	#4. Use cv2.addWeighted() to combine the results
	gray = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
	#5. Convert each pixel to unint8, then apply threshold to get binary image	
	gray = cv2.convertScaleAbs(gray)
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

def perspective_transform(image, verbose=False):
	"""
	Get bird's eye view from input image
	"""
	#1. Visually determine 4 source points and 4 destination points
	src_pts = np.array([[480, 250], [680, 250], [809, 370], [292, 370]], np.float32)
	dst_pts = np.array([[292, 0], [809, 0], [809, 370], [292, 370]], np.float32)

	#2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)
	Minv = np.linalg.inv(M)

	#3. Generate warped image in bird view using cv2.warpPerspective()
	warped_img = cv2.warpPerspective(image, M, (1242, 375))

	## TODO

	####

	return warped_img, M, Minv



if __name__ == '__main__':
	# # Import the image
	img = cv2.imread(r"test.png")
	cv2.imshow("image", img)
	
	#Test for the gradient thresh
	thresh_img = gradient_thresh(img)
	cv2.imshow("grad thresh", thresh_img)
	# cv2.waitKey(0)

	# Test for the color thresh
	thresh_img2 = color_thresh(img)
	cv2.imshow("color thresh", thresh_img2)

	# Test for the combined image
	img2 = combinedBinaryImage(img)
	cv2.imshow("combined", img2)
	# cv2.waitKey(0)

	# cv2.imshow("thresh img", thresh_img)
	wrp_img, M, Minv = perspective_transform(img2)
	cv2.imshow("transform", wrp_img)
	cv2.waitKey(0)













	