import cv2 as cv
import numpy as np

img = cv.imread('Cat.jpg')
img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
cv.imshow('original Cat', img)
cv.imshow('Histogram equalized', img_yuv)

cv.waitKey(0)