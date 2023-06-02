import cv2 as cv
import numpy as np

img = cv.imread(r'C:\Users\nazir\Documents\OpenCV\Images\box.jpg')
# img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

conners = cv.goodFeaturesToTrack(gray, 7, 0.05, 25)
conners = np.float32(conners)

for item in conners:
    x, y = item[0]
    print(x)
    cv.circle(img, (x,y), 5, 255, -1)
    

cv.imshow('Harris Corners', img)
cv.waitKey()