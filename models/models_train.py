"""
Perform Grid Search on Training Hyper Parameters for each architecture
"""

import time
import torch
import wandb
import warnings

from models.inception import inception_v3
from models.alexnet import alexnet
from models.cornet_s import cornet_s
from models.cornet_z import cornet_z
from models.mobilenet import mobilenet_v2
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn
from models.FaceNet import FaceNet
from models.LightCNN import LightCNN_V4
from models.DeepID_pytorch import deepID_1
from models.SphereFace import SphereFace

from utils.load_data import dataloader
from utils.train_test_functions import train_network, test_network, save_network_weights
from utils.arg_parser import get_config_parser


if __name__ == '__main__':

    parser = get_config_parser()
    args = parser.parse_args()
    start = time.time()
    # Check for the device
    if (args.device == "cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available, make that your environment is running on GPU "
            'Forcing device="cpu".'
        )
        args.device = "cpu"

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    dataset_loader, dataset_sizes = dataloader(args.batch_size, args.dataset)
    wandb.init(
            # set the wandb project where this run will be logged
            project="Modeling the Face Recognition System in the Brain",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "architecture": args.model,
            "dataset": args.dataset,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "num_classes": args.num_classes,
            "momentum": args.momentum, 
            "step_size": args.step_size,
            "gamma": args.gamma,
            "pretrained": args.pretrained,
            }
        )

    model_cls = {"alexnet": alexnet, "resnet18": resnet18, "resnet34": resnet34, "deepID_1": deepID_1, 
                "LightCNN": LightCNN_V4, "cornet_z": cornet_z, "cornet_s": cornet_s, "resnet50": resnet50,
                 "mobilenet": mobilenet_v2,  "vgg16_bn": vgg16_bn, 
                 "vgg19_bn": vgg19_bn, "inception_v3": inception_v3, "FaceNet": FaceNet, "SphereFace": SphereFace,
                 "resnet101": resnet101, "resnet152": resnet152, "vgg16": vgg16, "vgg19": vgg19}[args.model]
    
    model = model_cls(args.pretrained, args.num_classes, args.n_input_channels, args.weights)
    model.to(args.device)

    #Optimizer
    if args.optimizer == "adamw":
        optimizer_ft = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer_ft = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer_ft = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "momentum":
        optimizer_ft = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, args.step_size, args.gamma)

    Experiment_name=f"final_{args.model}_{args.num_classes}"

    model_ft, wandb = train_network(Experiment_name, model, criterion, optimizer_ft, exp_lr_scheduler, args.num_epochs, dataset_loader, dataset_sizes, wandb)
    acc=test_network(model_ft, dataset_loader, dataset_sizes)
    wandb.log ({"Test Acc": acc})

    print()
    time_training = time.time() - start
    print('Training ended in %s h %s m %s s' % (time_training // 3600, (time_training % 3600) // 60, time_training % 60))

    #  Save weights after training
    save_network_weights(model_ft, Experiment_name)