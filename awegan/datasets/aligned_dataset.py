import os.path as osp
from .image_folder import make_dataset
from .base_dataset import BaseDataset, get_params, get_transform
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = osp.join(opt.data_root, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(opt.load_size >= opt.crop_size), "the `crop_size` should be smaller than the `load_size`"
        BtoA = opt.direction == 'BtoA'
        self.input_nc = opt.output_nc if BtoA else opt.input_nc
        self.output_nc = opt.input_nc if BtoA else opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB_img = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w2, h = AB_img.size
        w = int(w2 / 2)
        A_img = AB_img.crop((0, 0, w, h))
        B_img = AB_img.crop((w, 0, w2, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A_img)
        B = B_transform(B_img)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
