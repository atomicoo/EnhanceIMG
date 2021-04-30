import numpy as np
import cv2


def easy_laplacian(img, ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_8U, ksize=ksize)

def easy_sobel(img, ksize=3, mode='xy'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = np.zeros(gray.shape, dtype=np.uint8)
    if 'x' in mode:
        sobel += cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=ksize)
    if 'y' in mode:
        sobel += cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=ksize)
    return sobel

def easy_scharr(img, mode='xy'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scharr = np.zeros(gray.shape, dtype=np.uint8)
    if 'x' in mode:
        scharr += cv2.Scharr(gray, cv2.CV_8U, 1, 0)
    if 'y' in mode:
        scharr += cv2.Scharr(gray, cv2.CV_8U, 0, 1)
    return scharr

def easy_canny(img, threshold=4):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, threshold, threshold*2.5)
