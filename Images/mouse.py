import cv2 as cv
import numpy as np



def detect_quadrant(event, x, y, flags, param):
    if event == cv.EVENT_FLAG_LBUTTON:
        if x > width/2:
            if y > height/2:
                point_top_left = (int(width/2), int(height/2))
                point_bottom_right = (width-1, height-1)
            else:
                point_top_left = (int(width/2), 0)
                point_bottom_right = (width-1, int(height/2))
        else:
            if y > height/2:
                point_top_left = (0, int(height/2))
                point_bottom_right = (int(width/2), height-1)
            else:
                point_top_left = (0, 0)
                point_bottom_right = (int(width/2), int(height/2))
        cv.rectangle(img, point_top_left, point_bottom_right, (0, 100, 0), -1)
        cv.rectangle(img, point_top_left, point_bottom_right,(0,100,0), -1)

if __name__=='__main__':
    width, height = 640, 480
    img = 255 * np.ones((height, width, 3), dtype=np.uint8)
    cv.namedWindow('Input window')
    cv.setMouseCallback('Input window', detect_quadrant)
    
    while True:
        cv.imshow('Input window', img)
        c = cv.waitKey(10)
        if c == 27:
            break
cv.destroyAllWindows()




