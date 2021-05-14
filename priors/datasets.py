import random
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt


##############################################################################
# Plotting Helper Function
##############################################################################

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow=8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    plt.xticks([]); plt.yticks([])
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid


##############################################################################
# Converting Helper Function
##############################################################################

def load_image(fpath, imsize=-1):
    """Load an image and resize to a specific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img_pil = Image.open(fpath)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if (imsize[0] != -1) and (img_pil.size != imsize):
        if imsize[0] > img_pil.size[0]:
            img_pil = img_pil.resize(imsize, Image.BICUBIC)
        else:
            img_pil = img_pil.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img_pil)

    return img_pil, img_np

def center_crop(img, imsize=64):
    """Center crop an image to a specific size."""

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if (imsize[0] < img.shape[1]) or (imsize[1] < img.shape[2]):

        bbox = [
            int((img.shape[1] - imsize[0])/2), 
            int((img.shape[2] - imsize[1])/2),
            int((img.shape[1] + imsize[0])/2),
            int((img.shape[2] + imsize[1])/2),
        ]

        img_cropped = img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
        img_cropped = np.ascontiguousarray(img_cropped)
    else:
        img_cropped = img

    return img_cropped

def random_crop(img, imsize=64):
    """Random crop an image to a specific size."""

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if (imsize[0] < img.shape[1]) or (imsize[1] < img.shape[2]):

        rand0 = random.randint(0, img.shape[1] - imsize[0])
        rand1 = random.randint(0, img.shape[2] - imsize[1])

        bbox = [
            rand0, rand1,
            rand0 + imsize[0], rand1 + imsize[1]
        ]

        img_cropped = img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
        img_cropped = np.ascontiguousarray(img_cropped)
    else:
        img_cropped = img

    return img_cropped

def pil_to_np(img_pil):
    """Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    """
    arr = np.array(img_pil)

    if len(arr.shape) == 3:
        arr = arr.transpose(2, 0, 1)
    else:
        arr = arr[None, ...]

    return arr.astype(np.float32) / 255.0

def np_to_pil(img_np): 
    """Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    """
    arr = np.clip(img_np*255, 0, 255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        arr = arr[0]
    else:
        arr = arr.transpose(1, 2, 0)

    return Image.fromarray(arr)

def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    """Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    """
    return img_var.detach().cpu().numpy()[0]


##############################################################################
# Processing Helper Function
##############################################################################

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def get_noisy_image(img_np, sigma=0.1):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np
