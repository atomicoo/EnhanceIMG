from colorspace import *
from filters import *
import cv2
import numpy as np
from . import multi_scale_retinex


def AttnMSR(img, sigma_list, threshold=10, a1=0.4, a2=0.5, b=0.5):
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
