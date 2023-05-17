import cv2 as cv
import numpy as np

img = cv.imread('Cat.jpg')
cv.imshow('Original Image', img)

# generate the kernel
kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
kernel_sharpen_2 = np.array([[1,1,1],[1,-7,1],[1,1,1]])
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
 [-1,2,2,2,-1],
 [-1,2,8,2,-1],
 [-1,2,2,2,-1],
 [-1,-1,-1,-1,-1]]) / 8.0
output_1 = cv.filter2D(img, -1, kernel_sharpen_1)
output_2 = cv.filter2D(img, -1, kernel_sharpen_2)
output_3 = cv.filter2D(img, -1, kernel_sharpen_3)
cv.imshow('sharpen Image 1', output_1)
cv.imshow('sharpen Image 2', output_2)
cv.imshow('sharpen Image 3', output_3)
cv.waitKey(0)