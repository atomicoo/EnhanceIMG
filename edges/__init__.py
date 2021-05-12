import os
import os.path as osp
import numpy as np
import cv2


def easy_laplacian(img, ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_8U, ksize=ksize)

def easy_LoG(img, ksize=3, sigma=0):
    img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_8U, ksize=ksize)

def easy_sobel(img, ksize=3, mode='xy'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelX = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=ksize)
    absX = cv2.convertScaleAbs(sobelX)
    sobelY = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=ksize)
    absY = cv2.convertScaleAbs(sobelY)
    if mode == 'x':
        return absX
    if mode == 'y':
        return absY
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

def easy_scharr(img, mode='xy'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scharrX = cv2.Scharr(gray, cv2.CV_8U, 1, 0)
    absX = cv2.convertScaleAbs(scharrX)
    scharrY = cv2.Scharr(gray, cv2.CV_8U, 0, 1)
    absY = cv2.convertScaleAbs(scharrY)
    if mode == 'x':
        return absX
    if mode == 'y':
        return absY
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

def easy_canny(img, threshold=32):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(img, threshold, threshold*2.5)

def easy_struct_forests(img):
    """Fast Edge Detection Using Structured Forests
    Python: https://github.com/ArtanisCV/StructuredForests
    Model : https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz
    """
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.float32(img) / 255.0
    pth = osp.join(osp.dirname(osp.abspath(__file__)), "struct-forests.yml.gz")
    retval  = cv2.ximgproc.createStructuredEdgeDetection(pth)
    return np.uint8(retval.detectEdges(img) * 255.0)
