import cv2 as cv
import numpy as np

haar_face_cascade = cv.CascadeClassifier(r'C:\Users\nazir\Documents\OpenCV\Videos\data\haarcascade_frontalface_alt.xml')
facemask = cv.imread(r'C:\Users\nazir\Documents\OpenCV\Videos\7pl11l_large.png')
h_facemask, w_face_mask = facemask.shape[:2]
print(w_face_mask)
if haar_face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

cap = cv.VideoCapture(0)
scaling_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # face_rects = haar_face_cascade.detectMultiScale(gray, 1.3, 5)
    face_rects = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in face_rects:
        if h > 0 and w > 0:
            h, w = int(1.4*h), int(1.0*w)
            y -= 0.1*h
            frame_rio = frame[y:y+h, x:x+w]
            face_mask_small = cv.resize(facemask, (w, h), interpolation=cv.INTER_AREA)

            gray_mask = cv.cvtColor(face_mask_small, cv.COLOR_BGR2GRAY)
            ret, mask =cv.threshold(gray_mask, 180, 255, cv.THRESH_BINARY_INV)
            mask_inv = cv.bitwise_not(mask)
            masked_face = cv.bitwise_and(face_mask_small, face_mask_small, mask=mask)
            masked_frame = cv.bitwise_and(frame_rio, frame_rio, mask=mask_inv)
            frame[y:y+h, x:x+w] = cv.add(masked_face, masked_frame)
    cv.imshow("Face Detector", frame)
    c = cv.waitKey(1)
    if c == 27:
        break

cap.release()
cv.destroyAllWindows()
