"""
Train CNN models
"""

import time
import torch.nn as nn
import torch.optim as optim

from models.alexnet import alexnet
from models.inception import inception_v3
from models.mobilenet import mobilenet_v2
from models.cornet_s import cornet_s
from models.vgg import vgg16, vgg19, vgg16_bn, vgg19_bn
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.deepface import DeepFace

from utils.load_data import dataloader

#path_plot = "/home/hamza97/scratch/plot/"
path_weights = "/home/hamza97/scratch/net_weights/"

dic = {1000: [30, 25, 12],
        500: [27, 13],
        300: [28, 14],
        150: [29, 15]
        }

def save_network_weights(model_ft,  file) :
    """Save the network after training"""
    state = model_ft.state_dict()
    torch.save(state,path_weights+file)


if __name__ == '__main__':
    # To work with GPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs=51
    learning_rate=[0.1, 0.01, 0.001] # 0.01 is the best learning rate for all tested models
    momentum=0.9
    step_size=10
    gamma=0.1
    pretrained=True
    num_classes_list=[1000, 500, 300, 150] # 1000 Gave the best performance across all HP and models
    # we also tested with 4000 and 10000 but no promising results.

    #model
    model_name="alexnet"
    if model_name in ["alexnet", "resnet18" "resnet34"]:
        batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    elif model_name in ["resnet50", "mobilenet"]:
        batch_sizes = [8, 16, 32, 64, 128, 256]
    elif model_name in ["resnet101", "resnet152", "vgg16_bn"]:
        batch_sizes = [8, 16, 32, 64, 128]
    elif model_name in ["resnet101", "resnet152"]:
        batch_sizes = [8, 16, 32]

    start = time.time()

    # Loss function
    criterion = nn.CrossEntropyLoss()
    best_acc=-1.0
    for num_classes in num_classes_list:
        for num_pictures in dic[num_classes]:
            for batch_size in batch_sizes:
                for lr in learning_rate:
                    if model_name == "inception_v3":
                        model = inception_v3(pretrained, progress=False, num_classes=num_classes)
                    elif model_name == "alexnet":
                        model = alexnet(pretrained, progress=False, num_classes=num_classes)
                    elif model_name ==  "cornet_s":
                        model = cornet_s(pretrained, num_classes=num_classes)
                    elif model_name == "mobilenet":
                        model = mobilenet_v2(pretrained, num_classes=num_classes)
                    elif model_name == "resnet18":
                        model = resnet18(num_classes, pretrained)
                    elif model_name == "resnet34":
                        model = resnet34(num_classes, pretrained)
                    elif model_name == "resnet50":
                        model = resnet50(num_classes, pretrained)
                    elif model_name == "resnet101":
                        model = resnet101(num_classes, pretrained)
                    elif model_name == "resnet152":
                        model = resnet152(num_classes, pretrained)
                    elif model_name == "vgg16":
                        model = vgg16(pretrained, progress=False, num_classes=num_classes)
                    elif model_name == "vgg16_bn":
                        model = vgg16_bn(pretrained, progress=False, num_classes=num_classes)
                    elif model_name == "vgg19":
                        model = vgg19(pretrained, progress=False, num_classes=num_classes)
                    elif model_name == "vgg19_bn":
                        model = vgg19_bn(pretrained, progress=False, num_classes=num_classes)

                    model.to(device)

                    # Optimizer
                    optimizer_ft = optim.SGD(model.parameters(), lr, momentum)

                    # Decay LR by a factor of 0.1 every 20 epochs
                    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size, gamma)

                    name="%s_%sLR_%sBatch_%s_%s"%(model_name, lr, batch_size, num_classes, num_pictures)
                    print("%%%%%%%%%% new model %%%%%%%%%%%%%%%%")
                    print(name)
                    # ##Training###
                    model_ft = train_model(name,
                                            batch_size,
                                            model,
                                            criterion,
                                            optimizer_ft,
                                            exp_lr_scheduler,
                                            num_epochs,
                                            num_classes,
                                            num_pictures)
                    acc=test_network(64, model_ft, num_classes, num_pictures)

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
    #save_network_weights(best_model,best_name)
