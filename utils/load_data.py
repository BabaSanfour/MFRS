import os
import torch
import torchvision
import sys
sys.path.append('../../MFRS')
from utils.config import study_path
data_path= os.path.join(study_path, 'hdf5/')
from torch.utils.data import Dataset
import h5py
import numpy as np


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


def dataloader(batch_n, dataset, num_classes=1000, num_pictures=30):
    """Return datasets train and valid"""
    # Argument :
    # batch_n : batch_size
    # Num classes : classes
    # Num pictures : pictures
    if dataset == "celebA":
        train_path = os.path.join(data_path, "new_train_%s_%s.h5"%(num_classes, num_pictures))
        valid_path = os.path.join(data_path, "new_valid_%s_%s.h5"%(num_classes, num_pictures))
        test_path = os.path.join(data_path, "new_test_%s_%s.h5"%(num_classes, num_pictures))
        mean, std = [0.3612], [0.3056]
    elif dataset == "VGGFace":
        train_path = os.path.join(data_path,"train.h5")
        valid_path = os.path.join(data_path,"valid.h5")
        test_path = os.path.join(data_path,"test.h5")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else :
        print('Error loading data')

    # extract mean and std:
    list_mean_std = mean_std[str(num_classes)+'_'+str(num_pictures)]
    mean, std = list_mean_std[0], list_mean_std[1]

    # ##Training dataset
    train_dataset = generate_Dataset_h5(train_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[mean], std=[std])]))

    # ##Validation dataset
    valid_dataset = generate_Dataset_h5(valid_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[mean], std=[std])]))

    test_dataset = generate_Dataset_h5(test_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[mean], std=[std])]))

    # ##Test dataset
    dataset_loader = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_n, num_workers=2, shuffle=True),
                      'valid': torch.utils.data.DataLoader(valid_dataset, batch_size=batch_n, num_workers=2, shuffle=True),
                      'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_n, num_workers=2, shuffle=True)}

    dataset_sizes = {'train': len(train_dataset), 'valid' : len(valid_dataset), 'test' : len(test_dataset)}

    return dataset_loader, dataset_sizes


def Stimuliloader(batch_n, file_name):
    """Return datasets train and valid"""
    # Argument :
    # batch_n : batch_size
    # file_name : file_name
    data=os.path.join(data_path, "%s.h5"%file_name)
    mean, std =[0.3612], [0.3056]

    # ##Stimuli dataset
    stimuli_dataset = generate_stimuli_h5(data,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[mean], std=[std])]))

    return torch.utils.data.DataLoader(stimuli_dataset, batch_size=batch_n,shuffle=False)
