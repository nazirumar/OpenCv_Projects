import cv2 as cv
import numpy as np
import sys

def overlay_vertical_seam(img, seam):
    img_seam_overlay = np.copy(img) 
    x_coords, y_coords = np.transpose([(i, int(j)) for i,j in enumerate(seam)])
    img_seam_overlay[x_coords, y_coords] = (0, 255, 0)
    return img_seam_overlay


def compute_energy_matrix(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    abs_sobel_x = cv.convertScaleAbs(sobel_x)
    abs_sobel_y = cv.convertScaleAbs(sobel_y)

    return cv.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)


def find_vertical_seam(img, energy):
    rows, cols = img.shape[:2]
    seam = np.zeros(img.shape[0])
    dist_to = np.zeros(img.shape[:2]) + sys.maxunicode
    dist_to[0,:] = np.zeros(img.shape[:1])
    edget_to = np.zeros(img.shape[:2])

    for row in range(rows-1):
        for col in range(cols):
            if col != 0:
                if dist_to[row+1, col-1] > dist_to[row, col] + energy[row+1, col-1]:
                    dist_to[row+1, col-1] = dist_to[row, col] + energy[row+1, col-1]
                    edget_to[row+1, col-1] = 1
            if dist_to[row+1, col] > dist_to[row, col] + energy[row+1, col]:
                dist_to[row+1, col] = dist_to[row, col] + energy[row+1, col]
                edget_to[row+1, col] = 0
            if col != cols - 1:
                if dist_to[rows+1, col+1] > dist_to[row, col] + energy[row+1, col+1]:
                    dist_to[row+1, col+1] = dist_to[row, col] + energy[row+1, col+1]
                    edget_to[row+1, col+1] = -1
    seam[rows-1] = np.argmin(dist_to[row-1, :])
    for i in (x for x in reversed(range(rows)) if x > 0):
        seam[i-1] = seam[i] + edget_to[i, int(seam[i])]
    return seam


def remove_vertical_seam(img, seam):
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(int(seam[row]), cols-1):
            img[row, col] = img[row, col+1]

    img = img[:, 0:cols-1]
    return img

if __name__ == '__main__':

    img_input = cv.imread(sys.argv[1])
    num_seam = int(sys.argv[2])

    img = np.copy(img_input)
    img_overlay_seam = np.copy(img_input)
    energy = compute_energy_matrix(img)

    for i in range(num_seam):
        seam = find_vertical_seam(img, energy)
        img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)
        img = remove_vertical_seam(img, seam)
        energy = compute_energy_matrix(img)
        print('Number of seams removed =', i+1)

    cv.imshow('Input', img_input)
    cv.imshow('Seams', img_overlay_seam)
    cv.imshow('Output', img)
    cv.waitKey()