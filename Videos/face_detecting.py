import cv2 as cv
import numpy as np

haar_face_cascade = cv.CascadeClassifier(r'C:\Users\nazir\Documents\OpenCV\Videos\data\haarcascade_frontalface_alt.xml')

cap = cv.VideoCapture(0)
scaling_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # face_rects = haar_face_cascade.detectMultiScale(gray, 1.3, 5)
    face_arects = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in face_arects:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    
    cv.imshow("Face Detector", frame)
    c = cv.waitKey(1)
    if c == 27:
        break

cap.release()
cv.destroyAllWindows()
