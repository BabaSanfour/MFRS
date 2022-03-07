import torch
import torchvision
from celebA_align_hdf5 import celebA_align_Dataset_h5

path_data='/home/hamza97/scratch/data/MFRS_data/hdf5/'

def dataloader_test(batch_n, num_classes, num_pictures):
    # ##Test dataset
    test_path=path_data+"test_%s_%s.h5"%(num_classes, num_pictures)
    test_dataset = celebA_align_Dataset_h5(test_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.485], std=[0.229])]))

    dataset_loader = {'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_n,shuffle=True)}

    dataset_sizes = {'test' : len(test_dataset)}

    return dataset_loader,dataset_sizes


def dataloader(batch_n , num_classes, num_pictures):
    """Return datasets train and valid"""
    # Argument :
    # batch_n : batch_size
    # Num classes : classes
    # Num pictures : pictures
    train_path=path_data+"train_%s_%s.h5"%(num_classes, num_pictures)
    valid_path=path_data+"valid_%s_%s.h5"%(num_classes, num_pictures)

    # ##Training dataset
    train_dataset = celebA_align_Dataset_h5(train_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.485], std=[0.229])]))

    # ##Validation dataset
    valid_dataset = celebA_align_Dataset_h5(valid_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.485], std=[0.229])]))



    dataset_loader = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_n,shuffle=True),
                      'valid': torch.utils.data.DataLoader(valid_dataset, batch_size=batch_n,shuffle=True)
                      }

    dataset_sizes = {'train': len(train_dataset), 'valid' : len(valid_dataset)}

    return dataset_loader,dataset_sizes
