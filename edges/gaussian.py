import math
import cv2
import numpy as np
from scipy import signal


def get_gaussian_kernel_1d(ksize, sigma=0):
    return cv2.getGaussianKernel(ksize=ksize, sigma=sigma)


def get_gaussian_kernel_2d(ksize, sigma=0):
    gaussian_kernel_1d = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    return gaussian_kernel_1d * gaussian_kernel_1d.T


def get_gaussian_kernel_2d_v2(ksize, sigma=0):
    gaussian_kernel_1d = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    return signal.convolve2d(gaussian_kernel_1d, gaussian_kernel_1d.T, mode='full', boundary='fill')


def get_LoG_kernel(ksize):
    """get Laplacian-of-Gaussian kernel.

    Ref: https://www.jianshu.com/p/762990148770
    """
    gau = get_gaussian_kernel_2d_v2(ksize=ksize-2)
    dif = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return signal.convolve2d(gau, dif, mode='full', boundary='fill')


def get_DoG_kernel(ksize, sigma=1.0):
    """get Difference-of-Gaussian kernel."""
    gau1 = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma*math.sqrt(2))
    gau2 = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma/math.sqrt(2))
    return gau1 - gau2


def get_Gabor_kernel(ksize, sigma=0.5, lambd=0.75):
    """get Gabor kernel.

    Ref: https://www.cnblogs.com/wojianxin/p/12574089.html
    Ref: https://blog.csdn.net/lhanchao/article/details/55006663
    """
    return cv2.getGaborKernel((ksize, ksize), sigma=sigma, theta=0, lambd=lambd, gamma=1.0)


if __name__ == '__main__':
    gray = cv2.imread("../Madison.png", cv2.IMREAD_GRAYSCALE)

    kernel_size = 7
    threshold = 0.5

    # kkk = get_LoG_kernel(kernel_size)
    # kkk = get_DoG_kernel(kernel_size)
    kkk = get_Gabor_kernel(kernel_size)

    conv = signal.convolve2d(gray, kkk, mode='same', boundary='fill')
    conv = (conv - np.min(conv)) / (np.max(conv) - np.min(conv))

    edge = np.zeros_like(conv, dtype=np.uint8)
    edge[conv > threshold] = 255
    edge[conv <= threshold] = 0

    cv2.imshow("EDGES", np.hstack([gray, edge]))
    cv2.waitKey(0)
