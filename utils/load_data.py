import torch
import torchvision
from utils.generate_data_from_hdf5 import generate_Dataset_h5, generate_stimuli_h5

path_data='../work/'
stimuli_path_data='/home/hamza97/scratch/data/MFRS_data/hdf5/'

mean_std = { '1000_30': [0.3612, 0.3056],
              '1000_25': [0.3736, 0.3082],
              '1000_12': [0.3642, 0.3047],
              '500_27': [0.3780, 0.3090],
              '500_14': [0.3620, 0.3041],
              '300_28': [0.3779, 0.3085],
              '300_14': [0.3614, 0.3045] }

def dataloader(batch_n, n_input_channels, num_classes=1000, num_pictures=30):
    """Return datasets train and valid"""
    # Argument :
    # batch_n : batch_size
    # Num classes : classes
    # Num pictures : pictures
    if n_input_channels == 1:
        train_path=path_data+"train_%s_%s.h5"%(num_classes, num_pictures)
        valid_path=path_data+"valid_%s_%s.h5"%(num_classes, num_pictures)
        test_path=path_data+"test_%s_%s.h5"%(num_classes, num_pictures)
        mean, std =[0.3612], [0.3056]
    elif n_input_channels == 3:
        train_path=path_data+"train.h5"
        valid_path=path_data+"valid.h5"
        test_path=path_data+"test.h5"
        mean, std= [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
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
    data=stimuli_path_data+"%s.h5"%file_name
    mean, std =[0.3612], [0.3056]

    # ##Stimuli dataset
    stimuli_dataset = generate_stimuli_h5(data,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[mean], std=[std])]))

    return torch.utils.data.DataLoader(stimuli_dataset, batch_size=batch_n,shuffle=False)
