from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path)
        w, h = A_img.size
        w8 = int(w / 8)
        a1 = A_img.crop((0, 0, w8, h))
        b1 = A_img.crop((w8, 0, 2 * w8, h))
        c1 = A_img.crop((2 * w8, 0, 3 * w8, h))
        A = Image.merge("RGB", (a1, b1, c1))
        d1 = A_img.crop((3 * w8, 0, 4 * w8, h))
        e1 = A_img.crop((4 * w8, 0, 5 * w8, h))
        f1 = A_img.crop((5 * w8, 0, 6 * w8, h))
        C = Image.merge("RGB", (d1, e1, f1))
        g1 = A_img.crop((6 * w8, 0, 7 * w8, h))
        h1 = A_img.crop((7 * w8, 0, w, h))
        D = Image.merge("RGB", (h1, h1, g1))


        A = self.transform(A)
        C = self.transform(C)
        D = self.transform(D)
        A = torch.cat((A, C, D))


        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
