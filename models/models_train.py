import os
import time
import torch
import wandb
import warnings
import logging
from typing import Optional, Any, Dict
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from inception import inception_v3
from cornet_s import cornet_s
from mobilenet import mobilenet_v2
from resnet import resnet50
from vgg import vgg16_bn
from FaceNet import FaceNet
from SphereFace import SphereFace

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.load_data import dataloader
from utils.arg_parser import get_training_config_parser
from utils.config import weights_path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_model(model: nn.Module, criterion, optimizer: Optimizer, scheduler: _LRScheduler, num_epochs: int, 
                dataset_loader: Dict[str, DataLoader], dataset_sizes: Dict[str, int], wandb: Optional[Any] = None) -> nn.Module:
    """
    Train the model using the train and validation datasets.

    Args:
        model (nn.Module): The neural network model to be trained.
        criterion: The loss function.
        optimizer: The optimizer for updating model weights.
        scheduler: The learning rate scheduler.
        num_epochs (int): Number of training epochs.
        dataset_loader (dict): Data loaders for train and validation sets.
        dataset_sizes (dict): Sizes of train and validation datasets.
        wandb: An optional WandB (Weights and Biases) logger for logging metrics.

    Returns:
        nn.Module: The best-trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = -1.0
    list_train_loss = []
    list_train_acc = []
    list_val_loss = []
    list_val_acc = []

    since = time.time()
    logger = logging.getLogger(__name__)

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        logger.info('-' * 10)

        n_correct = 0  # Correct predictions in the training set
        running_loss = 0.0  # Training loss

        model.train()  # Set the model to training mode

        for inputs, labels in dataset_loader["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)  # Forward pass
            n_correct += (torch.max(outputs, 1)[1].view(labels.data.size()) == labels.data).sum().item()
            loss = criterion(outputs, labels)  # Calculate loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update optimizer learning rate

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()  # Update learning rate

        train_acc = 100.0 * n_correct / dataset_sizes["train"]
        train_loss = running_loss / dataset_sizes["train"]
        logger.info(f'Train Loss: {train_loss:.2f} Acc: {train_acc:.2f}%')
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)

        # Evaluate performance on the validation set
        model.eval()  # Set the model to evaluation mode

        n_dev_correct = 0
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in dataset_loader["valid"]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                n_dev_correct += (torch.max(outputs, 1)[1].view(labels.data.size()) == labels.data).sum().item()

                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

        valid_acc = 100.0 * n_dev_correct / dataset_sizes["valid"]
        valid_loss = running_loss / dataset_sizes["valid"]

        list_val_loss.append(valid_loss)
        list_val_acc.append(valid_acc)

        logger.info(f'Valid Loss: {valid_loss:.2f} Acc: {valid_acc:.2f}%')

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model

        if wandb:
            wandb.log({"Val Loss": valid_loss, "Val Acc": valid_acc, "Train Loss": train_loss, "Train Acc": train_acc})

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 3600} h {(time_elapsed % 3600) // 60} m {time_elapsed % 60} s')
    logger.info(f'Best val Acc: {best_acc:.2f}%')

    return best_model


def test_model(model: nn.Module, dataset_loader: Dict[str, DataLoader], dataset_sizes: Dict[str, int]) -> [float, float]:
    """
    Test the model on a test dataset.

    Args:
        model (nn.Module): The neural network model to be tested.
        dataset_loader (Dict[str, DataLoader]): Data loaders for the test dataset.
        dataset_sizes (Dict[str, int]): Sizes of the test dataset.

    Returns:
        float: The average test accuracy.
    """

    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct_predictions = 0
    correct_topk_predictions = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()  # Loss function

    for inputs, labels in dataset_loader['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        # Calculate top-5 accuracy
        topk = 5
        _, topk_predictions = outputs.topk(k=5, dim=1)
        topk_predictions = topk_predictions.t()
        labels_reshaped = labels.view(1, -1).expand_as(topk_predictions)
        topk_correct = (topk_predictions == labels_reshaped)
        flattened_topk_correct = topk_correct.reshape(-1).float()
        correct_topk_predictions += flattened_topk_correct.float().sum(dim=0, keepdim=True)

        correct_predictions += (torch.max(outputs, 1)[1].view(labels.data.size()) == labels.data).sum().item()

        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

    avg_test_accuracy = 100.0 * correct_predictions / dataset_sizes["test"]
    avg_test_loss = test_loss / dataset_sizes["test"]
    avg_test_accuracy_topk = (100.0 * correct_topk_predictions / dataset_sizes["test"]).to("cpu").numpy().item()

    # Log the results
    logger.info(f'Test Loss: {avg_test_loss:.2f}')
    logger.info(f'Test Accuracy (Top1): {avg_test_accuracy:.2f}%')
    logger.info(f'Test Accuracy (Top5): {avg_test_accuracy_topk:.2f}%')

    return avg_test_accuracy, avg_test_accuracy_topk


def save_network_weights(model: nn.Module, weights_name: str) -> None:
    """
    Save the weights of a neural network model to a file.

    Args:
        model (nn.Module): The neural network model.
        weights_name (str): The name of the weights file.
    """
    try:
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(weights_path, weights_name))
        logger.info(f"Model weights saved to {weights_name}")
    except Exception as e:
        logger.error(f"Failed to save model weights: {e}")

if __name__ == '__main__':

    parser = get_training_config_parser()
    args = parser.parse_args()
    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    dataset_loader, dataset_sizes = dataloader(args.batch_size, args.dataset, args.analysis_type)
    
    # Initialize Weights and Biases
    wandb.init(
        project="Modeling the Face Recognition System in the Brain",
        config={
            "architecture": args.model,
            "dataset": args.dataset,
            "epochs": args.num_epochs,
            "num_classes": args.num_classes,
        }
    )

    # Choose the model class based on args.model
    model_cls = { "cornet_s": cornet_s, "resnet50": resnet50,
                 "mobilenet": mobilenet_v2,  "vgg16_bn": vgg16_bn, 
                "inception_v3": inception_v3, "FaceNet": FaceNet, 
                "SphereFace": SphereFace}[args.model]

    # Initialize the model
    model = model_cls(args.pretrained, args.num_classes, args.n_input_channels, args.transfer, args.in_weights)
    model.to(device)

    # Choose optimizer based on args.optimizer
    if args.optimizer == "adamw":
        optimizer_ft = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer_ft = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer_ft = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer_ft = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, args.step_size, args.gamma)

    experiment_name = f"final_{args.model}_{args.num_classes}"

    model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, args.num_epochs, dataset_loader, dataset_sizes, wandb)
    acc = test_model(model_ft, dataset_loader, dataset_sizes)
    wandb.log({"Test Acc": acc})

    logger.info("Training complete.")

    time_training = time.time() - start
    logger.info('Training ended in %s h %s m %s s' % (time_training // 3600, (time_training % 3600) // 60, time_training % 60))

    # Save weights after training
    save_network_weights(model_ft, args.out_weights)
