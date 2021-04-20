import cv2

def bgr2rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def rgb2bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def bgr2gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def bgr2hsv(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def hsv2bgr(img_hsv):
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

def bgr2hls(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)

def hls2bgr(img_hls):
    return cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR)

def bgr2lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

def lab2bgr(img_lab):
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

def bgr2luv(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LUV)

def luv2bgr(img_luv):
    return cv2.cvtColor(img_luv, cv2.COLOR_LUV2BGR)

def bgr2yuv(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)

