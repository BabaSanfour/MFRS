import sys
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

sys.path.append('/home/hamza97/MFRS/utils')
from celebA_align_hdf5 import celebA_align_Dataset_h5

path_data='/home/hamza97/scratch/data/MFRS_data/hdf5/'

dic = {1000: [30, 25, 12],
        500: [27, 13],
        300: [28, 14],
        150: [29, 15]}

def dataloader(batch_n , num_classes, num_pictures):
    """Return datasets train and valid"""
    # Argument :
    # batch_n : batch_size
    # Num classes : classes
    # Num pictures : pictures
    train_path=path_data+"train_%s_%s.h5"%(num_classes, num_pictures)
    valid_path=path_data+"valid_%s_%s.h5"%(num_classes, num_pictures)
    test_path=path_data+"test_%s_%s.h5"%(num_classes, num_pictures)

    # ##Training dataset
    train_dataset = celebA_align_Dataset_h5(train_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    # ##Validation dataset
    valid_dataset = celebA_align_Dataset_h5(valid_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    test_dataset = celebA_align_Dataset_h5(test_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    # ##Test dataset
    dataset_loader = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_n),
                      'valid': torch.utils.data.DataLoader(valid_dataset, batch_size=batch_n),
                      'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_n)}

    return dataset_loader

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    # Go over three sets of data
    for set in ['train', 'valid', 'test']:
        for data, _ in dataloader[set]:
            # Mean over batch, height and width, but not over the channels
            channels_sum += torch.mean(data)
            channels_squared_sum += torch.mean(data**2)
            num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

if __name__ == '__main__':

    for num_classes in [1000, 500, 300, 150]:
        for num_pictures in dic[num_classes]:
            dataset_loader = dataloader(64, num_classes, num_pictures)
            mean, std = get_mean_and_std(dataset_loader)
            print('Dataset %s class, %s picture:' % (num_classes ,num_pictures))
            print('Mean: %s , Std: %s' % (mean ,std))