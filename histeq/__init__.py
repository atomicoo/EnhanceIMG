import numpy as np
import cv2


def histogram_equalization(img):
    channels = cv2.split(img)
    for i in range(len(channels)):
        channels[i] = cv2.equalizeHist(channels[i])
    return cv2.merge(channels)

def contrast_limited_ahe(img, clipLimit=40.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    channels = cv2.split(img)
    for i in range(len(channels)):
        channels[i] = clahe.apply(channels[i])
    return cv2.merge(channels)

def adjust_gamma(img, gamma=1.0):
    channels = cv2.split(img)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    # apply gamma correction using the lookup table
    for i in range(len(channels)):
        channels[i] = cv2.LUT(np.array(channels[i], dtype=np.uint8), table)
    return cv2.merge(channels)
