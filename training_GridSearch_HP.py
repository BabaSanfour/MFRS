"""
Perform Grid Search on Training Hyper Parameters for each architecture
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from models.alexnet import alexnet
from models.inception import inception_v3
from models.mobilenet import mobilenet_v2
from models.cornet_s import cornet_s
from models.vgg import vgg16, vgg19, vgg16_bn, vgg19_bn
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.FaceNet import FaceNet
from models.LightCNN import LightCNN_V4
from models.DeepID_pytorch import deepID_1
from models.SphereFace import SphereFace

from utils.load_data import dataloader
from utils.train_test_functions import train_network, test_network, save_network_weights

dic = {1000: [30, 25, 12],
       500: [27, 13],
       300: [28, 14],
       150: [29, 15]
    }


path_weights = "/home/hamza97/scratch/net_weights/"

if __name__ == '__main__':
    # To work with GPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs=51
    learning_rate=[0.1, 0.01, 0.001] # 0.01 is the best learning rate for all tested models
    momentum=0.9
    step_size=10
    gamma=0.1
    pretrained=True
    n_input_channels=1 
    num_classes_list=[1000, 500, 300] # 1000 Gave the best performance across all HP and models
    # we also tested with 4000 and 10000 but no promising results.
    print('hamza')
    #model
    model_name="alexnet"
    # select batch sizes: we some of the models couldn't work with big batch sizes because of our gpu
    if model_name in ["alexnet", "resnet18" "resnet34", "deepID_1", "LightCNN"]:
        batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    elif model_name in ["cornet_s", "resnet50", "mobilenet"]:
        batch_sizes = [8, 16, 32, 64, 128, 256]
    elif model_name in ["resnet101", "resnet152", "vgg16_bn", "vgg19_bn"]:
        batch_sizes = [8, 16, 32, 64, 128]
    elif model_name in ["inception_v3", "FaceNet", "SphereFace"]:
        batch_sizes = [8, 16, 32, 64]
    elif model_name in ["resnet101", "resnet152", "vgg16", "vgg19"]:
        batch_sizes = [8, 16, 32]

    start = time.time()

    # Loss function
    criterion = nn.CrossEntropyLoss()
    best_acc=-1.0
    for num_classes in num_classes_list:
        for num_pictures in dic[num_classes]:
            for batch_size in batch_sizes:
                dataset_loader, dataset_sizes = dataloader(batch_size, n_input_channels)
                for lr in learning_rate:
                    if model_name == "inception_v3":
                        model = inception_v3(pretrained, num_classes, n_input_channels)
                    elif model_name == "alexnet":
                        model = alexnet(pretrained, num_classes, n_input_channels)
                    elif model_name ==  "cornet_s":
                        model = cornet_s(pretrained, num_classes, n_input_channels)
                    elif model_name == "mobilenet":
                        model = mobilenet_v2(pretrained, num_classes, n_input_channels)
                    elif model_name == "resnet18":
                        model = resnet18(pretrained, num_classes, n_input_channels)
                    elif model_name == "resnet34":
                        model = resnet34(pretrained, num_classes, n_input_channels)
                    elif model_name == "resnet50":
                        model = resnet50(pretrained, num_classes, n_input_channels)
                    elif model_name == "resnet101":
                        model = resnet101(pretrained, num_classes, n_input_channels)
                    elif model_name == "resnet152":
                        model = resnet152(pretrained, num_classes, n_input_channels)
                    elif model_name == "vgg16":
                        model = vgg16(pretrained, num_classes, n_input_channels)
                    elif model_name == "vgg16_bn":
                        model = vgg16_bn(pretrained, num_classes, n_input_channels)
                    elif model_name == "vgg19":
                        model = vgg19(pretrained, num_classes, n_input_channels)
                    elif model_name == "vgg19_bn":
                        model = vgg19_bn(pretrained, num_classes, n_input_channels)
                    elif model_name == "FaceNet":
                        model = FaceNet(pretrained, num_classes, n_input_channels)
                    elif model_name == "LightCNN":
                        model = LightCNN_V4(pretrained, num_classes, n_input_channels)
                    elif model_name == "deepID_1":
                        model = deepID_1(pretrained, num_classes, n_input_channels)
                    elif model == "SphereFace":
                        model = SphereFace(pretrained, num_classes, n_input_channels)

                    print('hamza2')
                    model.to(device)

                    # Optimizer
                    optimizer_ft = optim.SGD(model.parameters(), lr, momentum)

                    # Decay LR by a factor of 0.1 every 20 epochs
                    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size, gamma)

                    name="%s_%sLR_%sBatch_%s_%s"%(model_name, lr, batch_size, num_classes, num_pictures)
                    print("%%%%%%%%%% new model %%%%%%%%%%%%%%%%")
                    print(name)

                    ###Training & Validation###
                    model_ft = train_network(name, model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, dataset_loader, dataset_sizes)
                    ###Testing###
                    acc=test_network(model_ft, dataset_loader, dataset_sizes)

                    if acc>best_acc:
                        best_model=model_ft
                        best_name=name
                        best_acc=acc
                    print('=' * 10)

    print()
    time_training = time.time() - start
    print('Training ended in %s h %s m %s s' % (time_training // 3600, (time_training % 3600) // 60, time_training % 60))

    #  Save weights after training
    print('best model: ', best_name)
    print('best model accuracy: ', best_acc)
    # save_network_weights(best_model,best_name)
