"""
Train Vgg_16_bn
"""

import os
import time
import glob
import copy
from array import array

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from model import vgg16_bn
from celebA_align_hdf5 import celebA_align_Dataset_h5


def dataloader_test(batch_n, path):
    # ##Test dataset
    test_path=path+"data/hdf5/final/test/test_5000_many_224_final_greyscale.h5"
    test_dataset = celebA_align_Dataset_h5(test_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.485], std=[0.229])]))

    dataset_loader = {'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_n,shuffle=True)}

    dataset_sizes = {'test' : len(test_dataset)}

    return dataset_loader,dataset_sizes


def dataloader(batch_n , path):
    """Return datasets train and valid"""
    # Argument :
    # batch_n : batch_size
    # Num classes : classes
    # ##Training dataset
    train_path=path+"data/hdf5/final/train/train_20000_many_224_final_greyscale.h5"
    valid_path=path+"data/hdf5/final/valid/valid_5000_many_224_final_greyscale.h5"
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

def train_model(name, batch_size, path, model, criterion, optimizer, scheduler, num_epochs=25):
    """Train the model using the train and validation datasets"""

    best_acc = -1.0
    dataset_loader, dataset_sizes = dataloader(batch_size, path)

    list_trainLoss = []
    list_trainAcc = []
    list_valLoss = []
    list_valAcc = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch %s/%s' % (epoch + 1, num_epochs))
        print('-' * 10)

        #initialization
        n_correct = 0 #correct predictions train
        running_loss = 0.0 # train loss
        loss = 0.0
        # track history in train
        with torch.set_grad_enabled(True):
            # switch model to training mode,
            model.train();
            # batches_count_train = len(dataset_loader["train"])
            for inputs_, labels_ in dataset_loader["train"]:

                # Move data to work with GPU
                inputs = inputs_.to(device)
                labels = labels_.to(device)

                # zero the parameter gradients, clear gradient accumulators
                optimizer.zero_grad()
                # forward pass
                outputs = model(inputs)
                # calculate accuracy of predictions in the current batch
                n_correct += (torch.max(outputs, 1)[1].view(labels.data.size()) == labels.data).sum().item()
                # calculate loss of the network output with respect to training labels
                loss = criterion(outputs, labels)

                # backpropagate and update optimizer learning rate
                loss.backward();
                optimizer.step()

                #statictis
                running_loss += loss.item() * inputs.size(0)

            #update learning rate
            scheduler.step()
            # epoch loss and acc
            train_acc = 100. * n_correct/dataset_sizes["train"]
            train_loss = running_loss / dataset_sizes["train"]
            print('%s Loss: %.2f Acc: %.2f%%' % ("Train", train_loss, train_acc))
            list_trainLoss.append(train_loss)
            list_trainAcc.append(train_acc)

        # evaluate performance on validation set
        with torch.no_grad():
            # switch model to evaluation mode
            model.eval();

            # calculate accuracy on validation set
            n_dev_correct=0
            running_loss = 0.0

            loss = 0.0
            for inputs_, labels_ in dataset_loader["valid"]:
                # To work with GPU
                inputs = inputs_.to(device)
                labels = labels_.to(device)
                outputs = model(inputs)
                n_dev_correct += (torch.max(outputs, 1)[1].view(labels.data.size()) == labels.data).sum().item()

                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            valid_acc = 100. * n_dev_correct/dataset_sizes["valid"]
            valid_loss = running_loss / dataset_sizes["valid"]

            list_valLoss.append(valid_loss)
            list_valAcc.append(valid_acc)

            print('%s Loss: %.2f Acc: %.2f%%' % ("Valid", valid_loss, valid_acc))
            if valid_acc> best_acc:
                best_acc= valid_acc
                best_model = model
    training_results_path = path+'training_results/'+name
    output_file = open(training_results_path, 'wb')
    float_array = array('d', list_valAcc)
    float_array.tofile(output_file)
    float_array = array('d', list_valLoss)
    float_array.tofile(output_file)
    output_file.close()

    fig_path = path+'plot/clean_plot/%s.png'%(name)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

    ax1.plot(list_trainLoss, "--", label='train loss')
    ax2.plot(list_trainAcc,"--", label='train accuracy')
    ax1.plot(list_valLoss, label='valid loss')
    ax2.plot(list_valAcc, label='valid accuracy')
    ax2.legend()
    ax1.legend()
    fig.savefig(fig_path)
    fig.show()


    print()
    time_elapsed = time.time() - since
    print('Training complete in %s h %s m %s s' % (time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
    print('Best val Acc: %.2f%%' % (best_acc))

    return best_model

def test_network(batch_size, model_ft, path):

    # torch works with gpu
    model_ft.eval()

    dataset_loader, dataset_sizes = dataloader_test(batch_size, path)
    n_dev_correct=0
    test_loss = 0.0
    # The loss function
    criterion = nn.CrossEntropyLoss()

    for inputs_, labels_ in dataset_loader['test']:
        # To work with GPU
        inputs = inputs_.to(device)
        labels = labels_.to(device)
        outputs = model(inputs)
        n_dev_correct += (torch.max(outputs, 1)[1].view(labels.data.size()) == labels.data).sum().item()

        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

    avg_test_acc = 100. * n_dev_correct/dataset_sizes["test"]
    avg_test_loss = test_loss / dataset_sizes["test"]

    print('Test Loss: %.2f' % (avg_test_loss))
    print('Test Accuracy (Overall): %.2f%%' % (avg_test_acc))

    return avg_test_acc


def save_network_weights(model_ft, path, file) :
    """Save the network after training"""
    state = model_ft.state_dict()
    torch.save(state,path+'net_weights/final/'+file)


if __name__ == '__main__':
    # To work with GPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "/home/hamza97/projects/def-kjerbi/hamza97/grad_proj/"
    batch_sizes=[16, 128]
    num_epochs=50
    learning_rate=[0.01]
    momentum=0.9
    step_size=20
    gamma=0.1
    pretrained=True

    #model

    # Loss function
    criterion = nn.CrossEntropyLoss()
    best_acc=-1.0
    for batch_size in batch_sizes:
        for lr in learning_rate:
            model = vgg16_bn(pretrained)


            model.to(device)

            # Optimizer
            optimizer_ft = optim.SGD(model.parameters(), lr, momentum)
            #optimizer_ft = optim.Adam(deepface_model.parameters(), lr=0.01)

            # Decay LR by a factor of 0.1 every 20 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size, gamma)

            name="vgg16_bn_%sLR_%sBatch"%(lr, batch_size)
            print(name)
            # ##Training###
            model_ft = train_model(name,
                                    batch_size,
                                    path,
                                    model,
                                    criterion,
                                    optimizer_ft,
                                    exp_lr_scheduler,
                                    num_epochs)
            acc=test_network(64, model_ft, path)

            if acc>best_acc:
                best_model=model_ft
                best_name=name
                best_acc=acc
            print('=' * 10)

    #  Save weights after training
    print('best model: ', best_name)
    print('best model accuracy: ', best_acc)
    save_network_weights(best_model,path,best_name)
