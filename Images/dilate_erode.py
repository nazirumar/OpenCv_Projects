import cv2 as cv
import numpy as np

img = cv.imread('Cat.jpg')
kernel = np.ones((10,10), np.uint8)


img_erode = cv.erode(img, kernel, iterations=1)
img_dilate = cv.dilate(img, kernel, iterations=1)

cv.imshow('original', img)
cv.imshow('Erode', img_erode)
cv.imshow('Dilate', img_dilate)

cv.waitKey(0)
