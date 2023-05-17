import cv2 as cv
import numpy as np

img = cv.imread('Cat.jpg')
rows, cols = img.shape[:2]

# kernel_x = cv.getGaussianKernel(cols, 200)
# kernel_y = cv.getGaussianKernel(rows, 200)
kernel_x = cv.getGaussianKernel(int(1.5*cols), 200)
kernel_y = cv.getGaussianKernel(int(1.5*rows), 200)

kernel = kernel_y * kernel_x.T

mask = 255 * kernel / np.linalg.norm(kernel)
mask = mask[int(0.5*rows):, int(0.5*cols):]
output = np.copy(img)

for i in range(3):
    output[:, :, i] = output[:, :, i] * mask

cv.imshow('original', img)
cv.imshow('Viginate', output)
cv.waitKey(0)
