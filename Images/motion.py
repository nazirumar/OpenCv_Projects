import cv2 as cv
import numpy as np

img = cv.imread('Cat.jpg')
def motion_blur(size, frameName):
    kernel_motion_blue = np.zeros((size, size))
    kernel_motion_blue[int((size-1) / 2), :] = np.ones(size)
    kernel_motion_blue = kernel_motion_blue / size
    output = cv.filter2D(img, -1, kernel_motion_blue)
    cv.imshow(frameName, output)

if __name__ == '__main__':
    motion_blur(100, 'motion blur 100') 
    motion_blur(50, 'motion blur 50') 
    motion_blur(20, 'motion blur 20') 
    motion_blur(10, 'motion blur 10') 
    cv.waitKey(0)