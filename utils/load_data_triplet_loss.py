import os
import h5py
import numpy as np
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Resize,Compose
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import study_path

study_path_env = os.getenv('STUDY_PATH')
if study_path_env:
    study_path = study_path_env

data_path = os.path.join(study_path, 'hdf5/')

class HDF5Dataset(Dataset):
    """
    Dataset for reading triplet data (anchor, positive, negative) from an HDF5 file.

    Args:
        dir_path (str): Path to the HDF5 file.
        transform (callable, optional): A function/transform to apply to the data.
    """

    def __init__(self, dir_path, transform=None):
        self.file = h5py.File(dir_path, 'r')
        self.images = self.file['images']
        self.labels = self.file['meta'][:]
        self.transform = transform
        self.index = np.arange(len(self.labels))

    def __len__(self):
        """Number of images in the file."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return the triplet (anchor, positive, negative) and their associated labels."""
        anchor_img = self.images[idx]
        anchor_label = self.labels[idx]

        if self.transform:
            anchor_img = self.transform(anchor_img)

        pos_indices = self.index[(self.index != idx) & (self.labels == anchor_label)]
        if len(pos_indices) > 0:
            pos_idx = random.choice(pos_indices)
        else:
            pos_idx = idx  

        positive_img = self.images[pos_idx]
        if self.transform:
            positive_img = self.transform(positive_img)

        neg_indices = self.index[self.labels != anchor_label]
        if len(neg_indices) > 0:
            neg_idx = random.choice(neg_indices)
        else:
            neg_idx = idx  

        

        negative_img = self.images[neg_idx]
        if self.transform:
            negative_img = self.transform(negative_img)
        return anchor_img, positive_img, negative_img


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
        valid_loader = DataLoader(HDF5Dataset(os.path.join(data_path, valid_filename),
                                   transform=torchvision.transforms.Compose([ Resize((224,224)),ToTensor(),
                                                                             Normalize(mean=mean, std=std)])))
        valid_size = len(valid_loader)
        valid_size = int(valid_size * 0.5)  
        valid_dataset, test_dataset = random_split(valid_loader.dataset, [valid_size, valid_size])

        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

    else:
        raise ValueError("Invalid dataset name")

    train_loader = DataLoader(HDF5Dataset(os.path.join(data_path, train_filename),
                                            transform=torchvision.transforms.Compose([ToTensor(),
                                                                                    Normalize(mean=mean, std=std)])),
                                            batch_size=batch_size, num_workers=2, shuffle=True,drop_last=True)

  

    if dataset != "imagenet":
        valid_loader = DataLoader(HDF5Dataset(os.path.join(data_path, valid_filename),
                                          transform=torchvision.transforms.Compose([ToTensor(),
                                                                                    Normalize(mean=mean, std=std)])),
                                            batch_size=batch_size, num_workers=2, shuffle=False,drop_last=True)

        test_loader = DataLoader(HDF5Dataset(os.path.join(data_path, test_filename),
                                            transform=torchvision.transforms.Compose([ToTensor(),
                                                                                   Normalize(mean=mean, std=std)])),
                                            batch_size=batch_size, num_workers=2, shuffle=False,drop_last=True)

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
    data_path = "path_to_your_data"  
    data_file = os.path.join(data_path, f"{file_name}.h5")
    
   
    mean, std = (0.3612,), (0.3056,)
    stimuli_dataset = HDF5Dataset(data_file,
                                         transform=Compose([ToTensor(),
                                                            Normalize(mean=mean, std=std)]),
                                         has_labels=False)  

    stimuli_loader = DataLoader(stimuli_dataset, batch_size=batch_size, shuffle=False)

    return stimuli_loader

