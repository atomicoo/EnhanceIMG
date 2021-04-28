from colorspace import *
from filters import *
import cv2
import numpy as np
from . import multi_scale_retinex


def AttnMSR(img, sigma_list, threshold=10, a1=0.4, a2=0.5, b=0.5):
    """边缘注意力加权的增强多尺度Retinex算法
    """
    # TODO: 待完善

    H, L, S = cv2.split(bgr2hls(img))

    L_MSR = multi_scale_retinex(L.astype(np.float32), sigma_list)
    L_MSR = (L_MSR - np.min(L_MSR)) / (np.max(L_MSR) - np.min(L_MSR)) * 255.0
    L_MSR = np.uint8(np.minimum(np.maximum(L_MSR, 0), 255))

    # E = cv2.Canny(gaussian_filter(L, (3,3)), 100, 300)
    # E = cv2.Sobel(L, cv2.CV_8U, 1, 0) + cv2.Sobel(L, cv2.CV_8U, 0, 1)
    E = cv2.Laplacian(L, cv2.CV_8U, ksize=3)
    # cv2.imshow("Attn1", E)
    M = mean_filter(L, (5,5))

    A = np.zeros_like(L, dtype=np.float32)
    A[M<threshold] = a1
    A[M>=threshold] = a2
    W = A + b * E / 255.0
    W = mean_filter(W, (3,3))
    # cv2.imshow("Attn2", W)

    L_prime = (1-W)*L_MSR + W*L
    L_prime = np.uint8(np.minimum(np.maximum(L_prime, 0), 255))
    img_sobmsr = cv2.merge([H, L_prime, S])

    return hls2bgr(img_sobmsr)

def multi_scale_sharpen(img, w1=0.5, w2=0.5, w3=0.25, radius=3):
    """多尺度细节提升算法
    """

    img = np.float64(img)

    G1 = gaussian_filter(img, (radius,radius), 1.0)
    # cv2.imshow("Gau1", cv2.convertScaleAbs(G1))
    G2 = gaussian_filter(img, (radius*2-1,radius*2-1), 2.0)
    # cv2.imshow("Gau2", cv2.convertScaleAbs(G2))
    G3 = gaussian_filter(img, (radius*4-1,radius*4-1), 4.0)
    # cv2.imshow("Gau3", cv2.convertScaleAbs(G3))

    D1 = (1-w1*np.sign(img-G1)) * (img-G1)
    # cv2.imshow("Det1", D1)
    D2 = w2 * (G1-G2)
    # cv2.imshow("Det2", D2)
    D3 = w3 * (G2-G3)
    # cv2.imshow("Det3", D3)
    img_mss = img + D1 + D2 + D3
 
    return cv2.convertScaleAbs(img_mss)
