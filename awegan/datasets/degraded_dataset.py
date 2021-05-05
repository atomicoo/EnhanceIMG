import os
from .base_dataset import BaseDataset, get_transform
from .image_folder import make_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class DegradedDataset(BaseDataset):
    """This dataset class can load a set of natural images, and convert to degraded images with special function.
    """
    @staticmethod
    def modify_cmd_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (L) and
        the number of channels for output image is 2 (ab). The direction is from A to B
        """
        parser.set_defaults(degraded_mode='retinex')
        degraded_mode = parser.parse_known_args()[0].degraded_mode
        if degraded_mode == 'colorization':
            parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.data_root, opt.phase + opt.direction[0])
        self.AB_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
        assert(opt.direction == 'AtoB'), "the `direction` under dataset [DegradedDataset] should be only `AtoB`"
        self.transform = get_transform(opt, convert=False)
        self.degraded_mode = opt.degraded_mode

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)     -- the degraded of an image
            B (tensor)     -- the undegraded of an image
            A_paths (str)  -- image paths
            B_paths (str)  -- image paths (same as A_paths)
        """
        path = self.AB_paths[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        A, B = self.degraded_function(img)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def degraded_function(self, img):
        if self.degraded_mode == 'colorization':
            img = np.array(img)
            lab = color.rgb2lab(img).astype(np.float32)
            A = lab[[0], ...] / 50.0 - 1.0
            B = lab[[1, 2], ...] / 110.0

        elif self.degraded_mode == 'super_resolution':
            factor = 2
            B = np.array(img).astype(np.float32)
            lr_size = [img.size[0] // factor, img.size[1] // factor]
            img = img.resize(lr_size, Image.ANTIALIAS)
            A = np.array(img).astype(np.float32)
            A = A / 255.0 * 2.0 - 1.0
            B = B / 255.0 * 2.0 - 1.0

        elif self.degraded_mode == 'denoising':
            sigma = 50
            sigma_ = sigma / 255.0
            B = img = np.array(img).astype(np.float32)
            A = np.clip(img + sigma * np.random.normal(scale=sigma_, size=img.shape), 0, 255).astype(np.float32)
            A = A / 255.0 * 2.0 - 1.0
            B = B / 255.0 * 2.0 - 1.0

        elif self.degraded_mode == 'restoration':
            zero_fraction = 0.15
            B = img = np.array(img).astype(np.float32)
            img_mask = (np.random.random_sample(size=img.shape) > zero_fraction).astype(np.uint8)
            A = img * img_mask
            A = A / 255.0 * 2.0 - 1.0
            B = B / 255.0 * 2.0 - 1.0

        elif self.degraded_mode == 'retinex':
            img = np.array(img)
            A = img.astype(np.float32)
            img = retinex(img)
            B = img.astype(np.float32)
            A = A / 255.0 * 2.0 - 1.0
            B = B / 255.0 * 2.0 - 1.0
        
        elif self.degraded_mode == 'none':
            A = B = np.array(img).astype(np.float32)
            A = A / 255.0 * 2.0 - 1.0
            B = B / 255.0 * 2.0 - 1.0

        return A, B


# =================================================== #

import cv2

def replace_zeroes(data):
    """将所有零元素替换成非零最小元素，避免取对数时出现问题"""
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero

    return data

def single_scale_retinex(img, sigma):
    """单尺度Retinex算法：S(x,y)=R(x,y)*L(x,y)"""
    img = replace_zeroes(img)
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex

def multi_scale_retinex(img, sigma_list):
    """多尺度Retinex算法：在多个尺度上运用Retinex"""
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def simplest_color_balance(img, low_clip, high_clip):
    """简单白平衡：将RGB三通道的像素值分布压缩到同样的区间内"""
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):            
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c  
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img    

def MSRCP(img, sigma_list, low_clip, high_clip):
    """带色彩还原的多尺度视网膜增强算法 - GIMP"""
    img = np.float32(img) + 1.0
    intensity = np.sum(img, axis=2) / img.shape[2]
    retinex = multi_scale_retinex(intensity, sigma_list)
    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)
    intensity1 = simplest_color_balance(retinex, low_clip, high_clip)
    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0
    img_msrcp = np.zeros_like(img)
    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]
    img_msrcp = np.uint8(np.clip(img_msrcp - 1.0, 0.0, 255.0))
    return img_msrcp

def AttnMSR(img, sigma_list, threshold=10, a1=0.4, a2=0.5, b=0.5):
    """边缘注意力加权的增强多尺度Retinex算法"""
    H, L, S = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    L_MSR = multi_scale_retinex(L.astype(np.float32), sigma_list)
    L_MSR = (L_MSR - np.min(L_MSR)) / (np.max(L_MSR) - np.min(L_MSR)) * 255.0
    L_MSR = np.uint8(np.minimum(np.maximum(L_MSR, 0), 255))
    E = cv2.Laplacian(L, cv2.CV_8U, ksize=3)
    M = cv2.blur(L, ksize=(5, 5))
    A = np.zeros_like(L, dtype=np.float32)
    A[M<threshold] = a1
    A[M>=threshold] = a2
    W = A + b * E / 255.0
    W = cv2.blur(L, ksize=(3, 3))
    L_prime = (1-W) * L_MSR + W * L
    L_prime = np.uint8(np.minimum(np.maximum(L_prime, 0), 255))
    img_attnmsr = cv2.merge([H, L_prime, S])
    return cv2.cvtColor(img_attnmsr, cv2.COLOR_HSV2RGB)

def multi_scale_sharpen(img, w1=0.5, w2=0.5, w3=0.25, radius=3):
    """多尺度细节提升算法"""
    img = np.float64(img)
    G1 = cv2.GaussianBlur(img, (radius,radius), 1.0)
    G2 = cv2.GaussianBlur(img, (radius*2-1,radius*2-1), 2.0)
    G3 = cv2.GaussianBlur(img, (radius*4-1,radius*4-1), 4.0)
    D1 = (1-w1*np.sign(img-G1)) * (img-G1)
    D2 = w2 * (G1-G2)
    D3 = w3 * (G2-G3)
    img_mss = img + D1 + D2 + D3
    return cv2.convertScaleAbs(img_mss)

def retinex(img):
    # return AttnMSR(img, [15, 80, 250], 10)
    return MSRCP(img=img, sigma_list=[15, 80, 250],
                 low_clip=0.01, high_clip=0.99)
