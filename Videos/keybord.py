import argparse
import cv2 as cv
import numpy as np
def argument_parse():
    parser = argparse.ArgumentParser(description="change color space of the input video stream using keybord controls. The control  \
                                     keys are : \
                                     Grayscale - 'g', YU - 'y', HSV - 'h'")
    return parser


if __name__ == "__main__":
    args = argument_parse().parse_args()
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        raise OSError('Cannot open webcam')

cur_char = -1
prev_char = -1

while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, None, fx=0.9, fy=0.9, interpolation=cv.INTER_AREA)

        c = cv.waitKey(1)
        if c == 27:
            break

        if c > -1 and c != prev_char:
             cur_char = c
        prev_char = c

        if cur_char == ord('g'):
            output = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        elif cur_char == ord('y'):
            output = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
        elif cur_char == ord('h'):
            output = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        elif cur_char == ord('l'):
            kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
            [-1,2,2,2,-1],
            [-1,2,8,2,-1],
            [-1,2,2,2,-1],
            [-1,-1,-1,-1,-1]]) / 8.0
            output = cv.filter2D(frame,  -1, kernel_sharpen)
        else:
             output = frame
        cv.imshow('Webcam', output)

cap.release()
cv.destroyAllWindows()