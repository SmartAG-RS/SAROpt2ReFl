import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

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
        AB = Image.open(AB_path)

        # split AB image into A and B
        w, h = AB.size
        w8 = int(w / 9)
        a1 = AB.crop((0, 0, w8, h))
        b1 = AB.crop((w8, 0, 2 * w8, h))
        c1 = AB.crop((2 * w8, 0, 3 * w8, h))
        A = Image.merge("RGB", (a1, b1, c1))
        d1 = AB.crop((3 * w8, 0, 4 * w8, h))
        e1 = AB.crop((4 * w8, 0, 5 * w8, h))
        f1 = AB.crop((5 * w8, 0, 6 * w8, h))
        C = Image.merge("RGB", (d1, e1, f1))
        g1 = AB.crop((6 * w8, 0, 7 * w8, h))
        g2 = AB.crop((7 * w8, 0, 8 * w8, h))
        D = Image.merge("RGB", (g1, g1, g2))
        B = AB.crop((8 * w8, 0, w, h))
        B = Image.merge("RGB", (B, B, B))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, B.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        C = A_transform(C)
        D = A_transform(D)
        B = B_transform(B)
        A = torch.cat((A, C, D))

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
