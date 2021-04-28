import cv2


def mean_filter(img, kernel_size):
    """take mean value in the neighbourhood of center pixel.
    """
    return cv2.blur(img, ksize=kernel_size)

def median_filter(img, kernel_size):
    """take median value in the neighbourhood of center pixel.
    """
    return cv2.medianBlur(img, ksize=kernel_size)

def gaussian_filter(img, kernel_size, sigma=0):
    """take value weighted by pixel distance in the neighbourhood of center pixel.
    """
    return cv2.GaussianBlur(img, ksize=kernel_size, sigmaX=sigma, sigmaY=sigma)

def bilateral_filter(img, d=0, sigmaColor=40, sigmaSpace=10):
    """take value weighted by pixel+gray distance in the neighbourhood of center pixel.
    """
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

def joint_bilateral_filter(img, joint, d=0, sigmaColor=40, sigmaSpace=10):
    """joint bilateral filter.
    """
    return cv2.ximgproc.jointBilateralFilter(joint, img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

def guided_filter(img, guide, radius=33, eps=2, dDepth=-1):
    return cv2.ximgproc.guidedFilter(guide, img, radius=radius, eps=eps, dDepth=dDepth)

