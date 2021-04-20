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

