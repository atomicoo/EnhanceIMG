# %%
import cv2
from colorspace import *
from noises import *
from filters import *
from histeq import *
from retinex import *
from retinex.enhancer import *

# %%
img_bgr = cv2.imread("Madison.png")
cv2.imshow('BGR', img_bgr)

# %%
# img_rgb = bgr2rgb(img_bgr)
# # cv2.imshow('RGB', img_rgb)

# img_gray = bgr2gray(img_bgr)
# # cv2.imshow('GRAY', img_gray)

# img_hsv = bgr2hsv(img_bgr)
# # cv2.imshow('HSV', img_hsv)

# img_hls = bgr2hls(img_bgr)
# # cv2.imshow('HLS', img_hls)

# img_lab = bgr2lab(img_bgr)
# # cv2.imshow('LAB', img_lab)

# img_lab = bgr2lab(img_bgr)
# # cv2.imshow('LAB', img_lab)

# %%
# img_sp = add_sandp_noise(img_bgr)
# # cv2.imshow("SPN", img_sp)

# img_gs = add_gaussian_noise(img_bgr)
# cv2.imshow("GSN", img_gs)

# %%
# img_mn = mean_filter(img_sp, kernel_size=(3,3))
# # cv2.imshow("MNB-3", img_mn)
# img_mn = mean_filter(img_sp, kernel_size=(5,5))
# # cv2.imshow("MNB-5", img_mn)

# img_md = median_filter(img_sp, kernel_size=3)
# # cv2.imshow("MDB-3", img_md)
# img_md = median_filter(img_sp, kernel_size=5)
# # cv2.imshow("MDB-5", img_md)

# %%
# img_gf = gaussian_filter(img_bgr, kernel_size=(3,3), sigma=0)
# cv2.imshow("GSB-3", img_gf)
# img_gf = gaussian_filter(img_bgr, kernel_size=(5,5), sigma=0)
# cv2.imshow("GSB-5", img_gf)

# %%
# img_bf = cv2.bilateralFilter(img_gs, d=0, sigmaColor=40, sigmaSpace=10)
# cv2.imshow("BFX", img_bf)

# joint = gaussian_filter(img_bgr, kernel_size=(3,3), sigma=0)
# img_jb = joint_bilateral_filter(img_bgr, joint, d=0, sigmaColor=40, sigmaSpace=10)
# cv2.imshow("JBF", img_jb)

# %%
# img_gd = guided_filter(img_bgr, img_bgr, radius=33, eps=2, dDepth=-1)
# cv2.imshow("GDF", img_gd)

# %%
# img_he = histogram_equalization(img_bgr)
# cv2.imshow("HE", img_he)

# img_ahe = contrast_limited_ahe(img_bgr, clipLimit=255.0, tileGridSize=(8,8))
# cv2.imshow("AHE", img_ahe)

# img_clahe = contrast_limited_ahe(img_bgr, clipLimit=40.0, tileGridSize=(8,8))
# cv2.imshow("CLAHE", img_clahe)

# %%
# config = {
#     "sigma_list": [15, 80, 250],
#     "G"         : 5.0,
#     "b"         : 25.0,
#     "alpha"     : 125.0,
#     "beta"      : 46.0,
#     "low_clip"  : 0.01,
#     "high_clip" : 0.99
# }

# img_msrcr = MSRCR(
#         img_bgr,
#         config['sigma_list'],
#         config['G'],
#         config['b'],
#         config['alpha'],
#         config['beta'],
#         config['low_clip'],
#         config['high_clip']
#     )
# cv2.imshow("MSRCR", img_msrcr)

# img_amsrcr = automated_MSRCR(
#         img_bgr,
#         config['sigma_list']
#     )
# cv2.imshow("AMSRCR", img_amsrcr)

# img_msrcp = MSRCP(
#         img_bgr,
#         config['sigma_list'],
#         config['low_clip'],
#         config['high_clip']        
#     )
# cv2.imshow("MSRCP", img_msrcp)

# %%
sigma_list = [15, 80, 250]
img_attnmsr = AttnMSR(img_bgr, sigma_list, 10)
cv2.imshow("AttnMSR", img_attnmsr)
img_mss = multi_scale_sharpen(img_attnmsr)
cv2.imshow("AttnMSR+MSS", img_mss)

# %%
from matplotlib import pyplot as plt

color = ('b','g','r')
for i, cl in enumerate(color): 
    hist = cv2.calcHist([img_bgr], [i], None, [256], [0,256]) 
    plt.plot(hist, color=cl) 
    plt.xlim([0,256])
for i, cl in enumerate(color): 
    hist = cv2.calcHist([img_attnmsr], [i], None, [256], [0,256]) 
    plt.plot(hist, '--', color=cl) 
    plt.xlim([0,256])
plt.show()

# %%
cv2.waitKey()

# %%
# cv2.imwrite("enlighten.png", np.concatenate((img_bgr, img_attnmsr, img_mss), axis=1))
