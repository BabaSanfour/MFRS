import os
import h5py
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Resize

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import study_path
data_path= os.path.join(study_path, 'hdf5/')


class HDF5Dataset(Dataset):
    """
    Dataset for reading data from an HDF5 file.

    Args:
        dir_path (str): Path to the HDF5 file.
        transform (callable, optional): A function/transform to apply to the data.
        has_labels (bool, optional): Set to True if the dataset includes labels.
    """

    def __init__(self, dir_path, transform=None, has_labels=True):
        self.file = h5py.File(dir_path, 'r')
        dataset_shape = self.file['images'].shape
        self.n_images = dataset_shape[0]
        if len(dataset_shape) == 3:
            self.nx, self.ny = dataset_shape[1], dataset_shape[2]
        else:
            self.nx, self.ny, self.nz = dataset_shape[1], dataset_shape[2], dataset_shape[3]

        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        """Number of images in the file."""
        return self.n_images

    def __getitem__(self, idx):
        """Return the input image and optionally the associated label."""
        if len(self.file['images'].shape) == 3:
            input_h5 = self.file['images'][idx, :, :]
        else:
            input_h5 = self.file['images'][idx, :, :, :]

        sample = np.array(input_h5.astype('uint8'))

        if self.transform:
            sample = self.transform(sample)

        if self.has_labels:
            label_h5 = self.file['meta'][idx]
            label = torch.tensor(int(label_h5))
            return sample, label
        else:
            return sample


def dataloader(batch_size: int, dataset: str, analysis_type: str) -> (dict, dict):
    """
    Create data loaders for training, validation, and test datasets.

    Args:
        batch_size (int): The batch size for data loaders.
        dataset (str): The dataset name ('celebA', 'VGGFace', 'imagenet', etc.).
        analysis_type (str): The type of analysis being performed.

    Returns:
        dict: A dictionary containing data loaders for 'train', 'valid', and 'test' datasets.
        dict: A dictionary containing sizes of 'train', 'valid', and 'test' datasets.
    """

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if dataset == "celebA":
        train_filename = f"train_{analysis_type}_meg_stimuli.h5"
        valid_filename = f"valid_{analysis_type}_meg_stimuli.h5"
        test_filename = f"test_{analysis_type}_meg_stimuli.h5"
        mean, std = [0.3612], [0.3056]
    elif dataset == "VGGFace":
        train_filename = "train.h5"
        valid_filename = "valid.h5"
        test_filename = "test.h5"
    elif dataset == "imagenet":
        train_filename = f"imagenet_subset_train_{analysis_type}.h5"
        valid_filename = f"imagenet_subset_valid_{analysis_type}.h5"
        # Split 'valid' into 'valid' and 'test' for ImageNet
        valid_loader = DataLoader(HDF5Dataset(os.path.join(data_path, valid_filename),
                                   transform=torchvision.transforms.Compose([ Resize((224,224)),ToTensor(),
                                                                             Normalize(mean=mean, std=std)])))
        valid_size = len(valid_loader)
        valid_size = int(valid_size * 0.5)  # Splitting the 'valid' set in half
        valid_dataset, test_dataset = random_split(valid_loader.dataset, [valid_size, valid_size])

        # Create data loaders for validation and testing
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

    else:
        raise ValueError("Invalid dataset name")

    # Create data loaders
    train_loader = DataLoader(HDF5Dataset(os.path.join(data_path, train_filename),
                                            transform=torchvision.transforms.Compose([ToTensor(),
                                                                                    Normalize(mean=mean, std=std)])),
                                            batch_size=batch_size, num_workers=0, shuffle=True,drop_last=True)

    if dataset != "imagenet":
        valid_loader = DataLoader(HDF5Dataset(os.path.join(data_path, valid_filename),
                                          transform=torchvision.transforms.Compose([ToTensor(),
                                                                                    Normalize(mean=mean, std=std)])),
                                            batch_size=batch_size, num_workers=0, shuffle=False,drop_last=True)

        test_loader = DataLoader(HDF5Dataset(os.path.join(data_path, test_filename),
                                            transform=torchvision.transforms.Compose([ToTensor(),
                                                                                   Normalize(mean=mean, std=std)])),
                                            batch_size=batch_size, num_workers=0, shuffle=False,drop_last=True)

    # Create dictionaries for loaders and sizes
    data_loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    data_sizes = {'train': len(train_loader.dataset), 'valid': len(valid_loader.dataset), 'test': len(test_loader.dataset)}

    return data_loaders, data_sizes


def Stimuliloader(batch_size: int, file_name: str) -> DataLoader:
    """
    Create a data loader for stimuli data.

    Args:
        batch_size (int): The batch size for the data loader.
        file_name (str): The name of the HDF5 file containing stimuli data.

    Returns:
        DataLoader: A data loader for stimuli data.
    """
    data_file = os.path.join(data_path, f"{file_name}.h5")
    
    # Set mean and std for normalization
    mean, std = (0.3612,),(0.3056,)
    # Create a stimuli dataset
    stimuli_dataset = HDF5Dataset(data_file,
                                   transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                             torchvision.transforms.Normalize(mean=mean, std=std)]),
                                   has_labels=False)  

    # Create a data loader for the stimuli dataset
    stimuli_loader = DataLoader(stimuli_dataset, batch_size=batch_size, shuffle=False)

    return stimuli_loader

    
