import cv2 as cv
import numpy as np


def draw_rectangle(event, x, y, flags, params):
    global x_init, y_init, drawing, top_left_pt, bottom_right_pt

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            top_left_pt = (min(x_init,x), min(y_init, y))
            bottom_right_pt = (max(x_init, x), max(y_init, y))
            img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        top_left_pt = (min(x_init, x), min(y_init,y))
        bottom_right_pt = (max(x_init, x), max(y_init, y))
        img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]


if __name__ == '__main__':
    drawing = False
    top_left_pt, bottom_right_pt = (-1, -1), (-1,-1)

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    cv.namedWindow('Webcam')
    cv.setMouseCallback('Webcam', draw_rectangle)

    while True:
        ret, frame = cap.read()
        img = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        (x0, y0), (x1,y1) = top_left_pt, bottom_right_pt
        img[y0:y1, x0:x1] = 255 - img[y0:y1, x0:x1]
        cv.imshow("Webcam", img)

        c = cv.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv.destroyAllWindows()
