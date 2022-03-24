"""
Generate Face Dataset from hdf5 file
"""
# To do : regroup both aligh_df5f files in one file
#Pytorch
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class VGGFace_align_Dataset_h5(Dataset):
    """CelebA Dataset stored in hdf5 file"""
    def __init__(self, dir_path, transform=False):
        #read the hdf5 file
        self.file = h5py.File(dir_path, 'r')
        #print(self.file['images'].shape)
        self.n_images, self.nx, self.ny = self.file['images'].shape

        self.preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor();
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.augmentations = torchvision.transforms.Compose(
            [
                torchvision.transforms.ColorJitter(),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomAffine(
                    15, translate=None,
                    scale=None, shear=None,
                    resample=False, fillcolor=0
                ),
                torchvision.transforms.RandomPerspective(
                    distortion_scale=0.1, p=0.5, interpolation=Image.NEAREST
                ),
                torchvision.transforms.Resize(
                    self.shape,
                    interpolation=Image.NEAREST
                ),
                torchvision.transforms.RandomChoice(
                    [
                        torchvision.transforms.CenterCrop(self.shape[0] - k)
                        for k in range(0, int(
                        self.shape[0] * 0.05), 1)
                    ]
                ),
                torchvision.transforms.RandomGrayscale(p=0.1),
                torchvision.transforms.ToTensor()
            ]
        ) if augmentations else None

    def apply_augmentations(self, img):
        if self.augmentations is not None:
            return (torch.clamp(self.augmentations(img), 0, 1) - 0.5) * 2
        return (self.preprocessing(img) - 0.5) * 2

    def __len__(self):
        """number of images in the file"""
        return self.n_images

    def __getitem__(self, idx):
        """return the input image and the associated label"""
        input_h5 = self.file['images'][idx,:,:,:]
        label_h5 = self.file['meta'][idx]
        sample = np.array(input_h5.astype('uint8'))
        label = torch.tensor(int(label_h5))

        return self.apply_augmentations(sample), label
