"""
Generate Face Dataset from hdf5 file
"""

#Pytorch
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class generate_Dataset_h5(Dataset):
    """CelebA Dataset stored in hdf5 file"""
    def __init__(self, dir_path, transform=False):
        #read the hdf5 file
        self.file = h5py.File(dir_path, 'r')
        if len(self.file['images'].shape)==3:
            self.n_images, self.nx, self.ny = self.file['images'].shape
        else :
            self.n_images, self.nx, self.ny, self.nz = self.file['images'].shape

        self.transform = transform

    def __len__(self):
        """number of images in the file"""
        return self.n_images

    def __getitem__(self, idx):
        """return the input image and the associated label"""
        if len(self.file['images'].shape)==3:
            input_h5 = self.file['images'][idx,:,:]
        else:
            input_h5 = self.file['images'][idx,:,:,:]
        label_h5 = self.file['meta'][idx]
        sample = np.array(input_h5.astype('uint8'))
        label = torch.tensor(int(label_h5))
        sample = self.transform(sample)

        return sample, label


class generate_stimuli_h5(Dataset):
    """CelebA Dataset stored in hdf5 file"""
    def __init__(self, dir_path, transform=False):
        #read the hdf5 file
        self.file = h5py.File(dir_path, 'r')
        if len(self.file['images'].shape)==3:
            self.n_images, self.nx, self.ny = self.file['images'].shape
        else :
            self.n_images, self.nx, self.ny, self.nz = self.file['images'].shape

        self.transform = transform

    def __len__(self):
        """number of images in the file"""
        return self.n_images

    def __getitem__(self, idx):
        """return the input image and the associated label"""
        if len(self.file['images'].shape)==3:
            input_h5 = self.file['images'][idx,:,:]
        else:
            input_h5 = self.file['images'][idx,:,:,:]
        sample = np.array(input_h5.astype('uint8'))
        sample = self.transform(sample)

        return sample
