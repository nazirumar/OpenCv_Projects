import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)

# check if the webcam is opened correctly

if not cap.isOpened():
    raise IOError('Cannot open webcam')

kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
 [-1,2,2,2,-1],
 [-1,2,8,2,-1],
 [-1,2,2,2,-1],
 [-1,-1,-1,-1,-1]]) / 8.0

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=0.9, fy=0.9, interpolation=cv.INTER_AREA)
    sharpen = cv.filter2D(frame, -1, kernel_sharpen)
    cv.imshow('Webcam', sharpen)
    c = cv.waitKey(1)
    if c == 27:
        break

cap.release()
cv.destroyAllWindows()