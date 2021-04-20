import numpy as np
import random
import skimage


def add_sandp_noise(img, proportion=0.1):
    img = skimage.util.random_noise(img, mode='s&p', amount=proportion)
    return skimage.util.img_as_ubyte(img)

def add_gaussian_noise(img, mu=0, sigma=0.1):
    img = skimage.util.random_noise(img, mode='gaussian', mean=mu, var=sigma**2)
    return skimage.util.img_as_ubyte(img)


# def sp_noise(img, proportion=0.1):
#     height, width = img.shape[:2]
#     num = int(height * width * proportion)
#     for _ in range(num):
#         h = random.randint(0, height - 1)
#         w = random.randint(0, width - 1)
#         r = random.randint(0, 1) * 255
#         img[h, w] = (r, r, r)
#     return img

# def gaussian_noise(img, mu, sigma):
#     height, width = img.shape[:2]
#     for i in range(height):
#         for j in range(width):
#             g = random.gauss(mu, sigma)
#             r = np.where((g+img[i, j])>255, 255, (g+img[i, j]))
#             r = np.where(r<0, 0, r)
#             img[i, j] = np.round(r)

