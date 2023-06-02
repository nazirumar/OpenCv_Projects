import cv2 as cv
import numpy as np

img = cv.imread(r'C:\Users\nazir\Documents\OpenCV\Images\box.jpg')
img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

gray = np.float32(gray)

dst = cv.cornerHarris(gray, 2, 5, 0.04) # only
# dst = cv.cornerHarris(gray, 14, 5, 0.04) # soft

dst = cv.dilate(dst, None)

img[dst > 0.01*dst.max()] = [0,0,0]

cv.imshow('Harris Corners', img)
cv.waitKey()