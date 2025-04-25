import os
import h5py
import numpy as np
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Resize,Compose

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import study_path
data_path= os.path.join(study_path, 'hdf5/')



class HDF5Dataset(Dataset):
    def __init__(self, dir_path, transform=None, has_labels=True):
        self.file = h5py.File(dir_path, 'r')
        self.images = self.file['images']
        self.labels = self.file['meta'][:]
        self.transform = transform
        self.has_labels = has_labels
        self.index = np.arange(len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_anchor = self.images[idx]
        label_anchor = self.labels[idx]

        if self.transform:
            img_anchor = self.transform(img_anchor)

        if self.has_labels:
            pos_indices = self.index[(self.index != idx) & (self.labels == label_anchor)]
            if len(pos_indices) > 0:
                idx_pos = random.choice(pos_indices)
            else:
                idx_pos = idx

            img_positive = self.images[idx_pos]
            if self.transform:
                img_positive = self.transform(img_positive)

            neg_indices = self.index[self.labels != label_anchor]
            if len(neg_indices) > 0:
                idx_neg = random.choice(neg_indices)
            else:
                idx_neg = idx

            img_negative = self.images[idx_neg]
            if self.transform:
                img_negative = self.transform(img_negative)

            return (img_anchor, img_positive, img_negative, 
                    torch.tensor(label_anchor, dtype=torch.long))

        return img_anchor


def dataloader(batch_size: int, dataset: str, analysis_type: str) -> (dict, dict):
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
                                              transform=Compose([Resize((224, 224)), ToTensor(),
                                                                 Normalize(mean=mean, std=std)])))
        valid_size = len(valid_loader)
        valid_size = int(valid_size * 0.5)
        valid_dataset, test_dataset = random_split(valid_loader.dataset, [valid_size, valid_size])
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        raise ValueError("Invalid dataset name")

    train_loader = DataLoader(HDF5Dataset(os.path.join(data_path, train_filename),
                                          transform=Compose([ToTensor(),
                                                             Normalize(mean=mean, std=std)]),
                                          ),
                              batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)

    if dataset != "imagenet":
        valid_loader = DataLoader(HDF5Dataset(os.path.join(data_path, valid_filename),
                                              transform=Compose([ToTensor(),
                                                                 Normalize(mean=mean, std=std)]),
                                              ),
                                  batch_size=batch_size, num_workers=0, shuffle=False, drop_last=True)

        test_loader = DataLoader(HDF5Dataset(os.path.join(data_path, test_filename),
                                             transform=Compose([ToTensor(),
                                                                Normalize(mean=mean, std=std)]),
                                             ),
                                 batch_size=batch_size, num_workers=0, shuffle=False, drop_last=True)

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
    data_path = "path_to_your_data"  # Make sure to set this to your actual data path
    data_file = os.path.join(data_path, f"{file_name}.h5")
    
    # Set mean and std for normalization
    mean, std = (0.3612,), (0.3056,)
    # Create a stimuli dataset
    stimuli_dataset = HDF5Dataset(data_file,
                                   transform=Compose([ToTensor(),
                                                      Normalize(mean=mean, std=std)]),
                                   has_labels=False)  

    # Create a data loader for the stimuli dataset
    stimuli_loader = DataLoader(stimuli_dataset, batch_size=batch_size, shuffle=False)

    return stimuli_loader
