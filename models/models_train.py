import os
import time
import torch
import wandb
import random
import logging
import numpy as np
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

from Centerloss import CenterLoss, MLoss, CircleLossSoftplus, CircleLossExp, ContrastiveLoss, TripletLoss

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.load_data import dataloader
from utils.arg_parser import get_training_config_parser
from utils.config import weights_path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


checkpoint_dir = '/scratch/mariem12/net_weights'
outweights_dir = '/scratch/mariem12/net_weights'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(outweights_dir):
    os.makedirs(outweights_dir)


def init_weights_randomly(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

def get_loss_function(args, device):
   
    if args.loss_function == 'center_loss':
        criterion_ce = nn.CrossEntropyLoss()
        criterion_center = CenterLoss(num_classes=args.num_classes, feat_dim=512, device=device)
        def combined_loss(logits, labels, features):
            ce_loss = criterion_ce(logits.to(device), labels.to(device))
            center_loss_value = criterion_center(features.to(device), labels.to(device))
            scaled_center_loss = args.center_loss_weight * center_loss_value  # Scale the center loss
            
            combined_loss_value = ce_loss + scaled_center_loss  # Calculate combined loss

            return combined_loss_value, ce_loss, scaled_center_loss

        return combined_loss, criterion_center, criterion_ce
    elif args.loss_function in ['cosface', 'arcface', 'sphereface']:
        return MLoss(in_features=512, out_features=args.num_classes, loss_type=args.loss_function, s=args.s, m=args.m,device=device)
    elif args.loss_function == 'circle_softplus': 
        return CircleLossSoftplus(m=args.m, gamma=args.gamma)
    elif args.loss_function == 'circle_exp':
        return CircleLossExp(scale=args.scale, margin=args.margin, similarity=args.similarity)
    elif args.loss_function == 'ContrastiveLoss':
        return ContrastiveLoss(margin=args.margin)
    elif args.loss_function == 'TripletLoss':
        return TripletLoss(margin=args.margin)
    else:
        raise ValueError("Unsupported loss function")

def load_dataloader(args):
    if args.loss_function in ['center_loss', 'cosface', 'arcface', 'sphereface']:
        from utils.load_data import dataloader
    elif args.loss_function in ['circle_softplus', 'circle_exp']:
        from utils.load_data_circle_loss import dataloader
    elif args.loss_function in ['ContrastiveLoss']:
        from utils. load_data_constrastive_loss import dataloader
    elif args.loss_function in ['TripletLoss']:
        from utils. load_data_triplet_loss import dataloader       
    else:
        raise ValueError("Invalid loss function")

    return dataloader(args.batch_size, args.dataset, args.analysis_type)

def train_model(model: nn.Module, optimizer: Optimizer, scheduler: _LRScheduler, num_epochs: int, loss_func,
                dataset_loader: Dict[str, DataLoader], dataset_sizes: Dict[str, int], wandb: Optional[Any] = None) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.loss_function == 'center_loss':
        loss_func, criterion_center, criterion_ce = loss_func
        center_optimizer = torch.optim.SGD(criterion_center.parameters(), lr=args.center_lr)

    best_acc = -1.0
    list_train_loss = []
    list_train_acc = []
    list_val_loss = []
    list_val_acc = []
    since = time.time()

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        logger.info('-' * 10)
        n_correct = 0  # Correct predictions in the training set
        running_loss = 0.0  # Training loss
        running_ce_loss = 0.0
        running_center_loss = 0.0
        model.train()  # Set the model to training mode

        #for data in dataset_loader["train"]:
        for step, data in enumerate(dataset_loader["train"]): 
            optimizer.zero_grad()  # Zero the parameter gradients
            if args.loss_function == 'TripletLoss':
                anchor, positive, negative = data
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_features = model(anchor)
                positive_features = model(positive)
                negative_features = model(negative)
                loss = loss_func(anchor_features, positive_features, negative_features)

                loss.backward()  
                inputs = anchor

                pos_dist = torch.norm(anchor_features - positive_features, p=2, dim=1)
                neg_dist = torch.norm(anchor_features - negative_features, p=2, dim=1)
                n_correct += torch.sum(pos_dist < neg_dist).item()

                
            elif args.loss_function == 'ContrastiveLoss':
                img1, img_pos, label_pos, img_neg, label_neg = data
                img1, img_pos, img_neg = img1.to(device), img_pos.to(device), img_neg.to(device)
                label_pos, label_neg = label_pos.to(device), label_neg.to(device)
                
                feature1 = model(img1)
                feature_pos = model(img_pos)
                feature_neg = model(img_neg)
                
                pos_loss = loss_func(feature1, feature_pos, label_pos)
                neg_loss = loss_func(feature1, feature_neg, label_neg)
                loss = (pos_loss + neg_loss) / 2

                loss.backward()
                inputs = img1  

                pos_dist = torch.norm(feature1 - feature_pos, dim=1)
                neg_dist = torch.norm(feature1 - feature_neg, dim=1)
                n_correct += ((pos_dist < neg_dist).sum().item() + (neg_dist > pos_dist).sum().item())

            elif args.loss_function in ['circle_softplus', 'circle_exp']:
                img_anchor, img_positive, img_negative, labels = data
                img_anchor, img_positive, img_negative = img_anchor.to(device), img_positive.to(device), img_negative.to(device)
                labels = labels.to(device)

                features_anchor = model(img_anchor)
                features_positive = model(img_positive)
                features_negative = model(img_negative)

                features = torch.cat([features_anchor, features_positive, features_negative], dim=0)
                labels = torch.cat([labels, labels, labels], dim=0)
                loss = loss_func(features, labels)
                inputs = img_anchor  # Assign a value to 'inputs' for consistency
            
            
            elif args.loss_function =='center_loss':
                center_optimizer.zero_grad()  # Zero the parameter gradients for center optimizer
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits, features = model(inputs)
                n_correct += (logits.argmax(dim=1).view(-1) == labels.view(-1)).sum().item()
                loss, ce_loss, center_loss_value = loss_func(logits, labels, features)
                loss.backward()
                # add code
                for param in criterion_center.parameters():
                    param.grad.data *= (1.0 / args.center_loss_weight) 
                center_optimizer.step()  

                running_ce_loss += ce_loss.item() * inputs.size(0)
                running_center_loss += center_loss_value.item() * inputs.size(0)


            else:  
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits, features = model(inputs)
                loss = loss_func(features, labels)
                loss.backward()  
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping      
                n_correct += (logits.argmax(dim=1).view(-1) == labels.view(-1)).sum().item()
                
                

            

            optimizer.step()  # Update optimizer learning rate

            running_loss += loss.item()*inputs.size(0)

        scheduler.step()  # Update learning rate

        train_acc = 100.0 * n_correct /(dataset_sizes["train"])
        train_loss = running_loss / dataset_sizes["train"]
        train_ce_loss = running_ce_loss / dataset_sizes["train"]
        train_center_loss = running_center_loss / dataset_sizes["train"]
        logger.info(f'Epoch {epoch}/{num_epochs - 1} Train Loss: {train_loss:.4f} train_acc: {train_acc:.4f} train_ce_loss: {train_ce_loss:.4f} train_center_loss: {train_center_loss:.4f} %' )
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)

        # Evaluate performance on the validation set
        model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        n_dev_correct = 0
        running_ce_loss = 0.0
        running_center_loss = 0.0
        n_dev_correct = 0


        with torch.no_grad():
            for data in dataset_loader["valid"]:
                if args.loss_function == 'TripletLoss':
                    anchor, positive, negative = data
                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                    # Forward pass
                    anchor_features = model(anchor)
                    positive_features = model(positive)
                    negative_features = model(negative)

                    loss = loss_func(anchor_features, positive_features, negative_features)

                    pos_dist = torch.norm(anchor_features - positive_features, p=2, dim=1)
                    neg_dist = torch.norm(anchor_features- negative_features, p=2, dim=1)
                    n_dev_correct += torch.sum(pos_dist < neg_dist).item()
                    #_, preds = torch.max(anchor_output, 1)
                    #n_dev_correct += (preds == anchor).sum().item()

                    inputs = anchor 
                    running_loss += loss.item() * inputs.size(0)

                elif args.loss_function == 'ContrastiveLoss':
                    img1, img_pos, label_pos, img_neg, label_neg = data
                    img1, img_pos, img_neg = img1.to(device), img_pos.to(device), img_neg.to(device)
                    label_pos, label_neg = label_pos.to(device), label_neg.to(device)
                    
                    feature1 = model(img1)
                    feature_pos = model(img_pos)
                    feature_neg = model(img_neg)
                    
                    pos_loss = loss_func(feature1, feature_pos, label_pos)
                    neg_loss = loss_func(feature1, feature_neg, label_neg)
                    loss = (pos_loss + neg_loss) / 2

                    pos_dist = torch.norm(feature1 - feature_pos, dim=1)
                    neg_dist = torch.norm(feature1 - feature_neg, dim=1)
                    n_dev_correct += ((pos_dist < neg_dist).sum().item() + (neg_dist > pos_dist).sum().item())

                    inputs = img1 
                    running_loss += loss.item() * inputs.size(0)

                elif args.loss_function in ['circle_softplus', 'circle_exp']:
                    img_anchor, img_positive, img_negative, labels = data
                    img_anchor, img_positive, img_negative = img_anchor.to(device), img_positive.to(device), img_negative.to(device)
                    labels = labels.to(device)

                    features_anchor = model(img_anchor)
                    features_positive = model(img_positive)
                    features_negative = model(img_negative)

                    if isinstance(features_anchor, tuple):
                        features_anchor = features_anchor[0]
                    if isinstance(features_positive, tuple):
                        features_positive = features_positive[0]
                    if isinstance(features_negative, tuple):
                        features_negative = features_negative[0]

                    features = torch.cat([features_anchor, features_positive, features_negative], dim=0)

                    labels = torch.cat([labels, labels, labels], dim=0)
                    loss = loss_func(features, labels)

                    inputs = img_anchor 
                    running_loss += loss.item() * inputs.size(0)

                elif args.loss_function =='center_loss':
                    
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    logits, features = model(inputs)

                    n_dev_correct += (logits.argmax(dim=1).view(-1) == labels.view(-1)).sum().item()
                    loss, ce_loss, center_loss_value = loss_func(logits, labels, features)
                    
                    running_ce_loss += ce_loss.item() * inputs.size(0)
                    running_center_loss += center_loss_value.item() * inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                
                    
                else:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    logits, features = model(inputs)

                    n_dev_correct += (logits.argmax(dim=1).view(-1) == labels.view(-1)).sum().item()
                    loss = loss_func(features, labels)
                    running_loss += loss.item() * inputs.size(0)

                

        valid_acc = 100.0 * n_dev_correct /(dataset_sizes["valid"])
        valid_loss = running_loss / dataset_sizes["valid"]
        valid_ce_loss = running_ce_loss / dataset_sizes["valid"]
        valid_center_loss = running_center_loss / dataset_sizes["valid"]
        #valid_combined_loss = valid_ce_loss + valid_center_loss
        list_val_loss.append(valid_loss)
        list_val_acc.append(valid_acc)

        logger.info(f'Validation_Loss: {valid_loss :.4f} Validation_Acc: {valid_acc:.4f} valid_ce_loss:{valid_ce_loss:.4f} valid_center_loss:{valid_center_loss:.4f} %')

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model

        if wandb:
            wandb.log({
                "Epoch": epoch,
                #"Train Combined Loss": train_combined_loss,
                "Train Cross-Entropy Loss": train_ce_loss,
                "Train Center Loss": train_center_loss,
                "Train Loss": train_loss,
                "Train Acc": train_acc,
                #"Validation Combined Loss": valid_combined_loss,
                "Validation Cross-Entropy Loss": valid_ce_loss,
                "Validation Center Loss": valid_center_loss,
                "Val Loss": valid_loss,
                "Validation Acc": valid_acc,
            })
         # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_of_center_loss{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")


    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 3600} h {(time_elapsed % 3600) // 60} m {time_elapsed % 60} s')
    logger.info(f'Best val Acc: {best_acc:.2f}%')

    return best_model

def test_model(model: nn.Module, dataset_loader: Dict[str, DataLoader], dataset_sizes: Dict[str, int], loss_func) -> [float, float]:

    model.eval() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct_predictions = 0
    correct_topk_predictions = 0
    test_loss = 0.0
    ce_test_loss = 0.0
    center_test_loss = 0.0

    criterion_center = None
    criterion_ce = None

    if isinstance(loss_func, tuple):
        combined_loss, criterion_center, criterion_ce = loss_func
    else:
        combined_loss = loss_func


    with torch.no_grad():
        for data in dataset_loader['test']:
            if args.loss_function == 'TripletLoss':
                anchor, positive, negative = data
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_output = model(anchor)
                positive_output = model(positive)
                negative_output = model(negative)
                
                loss = loss_func(anchor_output, positive_output, negative_output)

                pos_dist = torch.norm(anchor_output - positive_output, p=2, dim=1)
                neg_dist = torch.norm(anchor_output- negative_output, p=2, dim=1)
                correct_predictions+= torch.sum(pos_dist < neg_dist).item()

                inputs = anchor  

                test_loss += loss.item() * inputs.size(0)


            elif args.loss_function == 'ContrastiveLoss':
                img1, img_pos, label_pos, img_neg, label_neg = data
                img1, img_pos, img_neg = img1.to(device), img_pos.to(device), img_neg.to(device)
                label_pos, label_neg = label_pos.to(device), label_neg.to(device)
                    
                feature1 = model(img1)
                feature_pos = model(img_pos)
                feature_neg = model(img_neg)
                    
                pos_loss = loss_func(feature1, feature_pos, label_pos)
                neg_loss = loss_func(feature1, feature_neg, label_neg)
                loss = (pos_loss + neg_loss) / 2


                # Calculate distances and determine correctness
                pos_dist = torch.norm(feature1 - feature_pos, dim=1)
                neg_dist = torch.norm(feature1 - feature_neg, dim=1)
                correct_predictions += ((pos_dist < neg_dist).sum().item() + (neg_dist > pos_dist).sum().item())

                inputs = img1  
                test_loss += loss.item() * inputs.size(0)
        

            elif args.loss_function in ['circle_softplus', 'circle_exp']:
                img_anchor, img_positive, img_negative, labels = data
                img_anchor, img_positive, img_negative = img_anchor.to(device), img_positive.to(device), img_negative.to(device)
                labels = labels.to(device)

                features_anchor = model(img_anchor)
                features_positive = model(img_positive)
                features_negative = model(img_negative)

                if isinstance(features_anchor, tuple):
                    features_anchor = features_anchor[0]
                if isinstance(features_positive, tuple):
                    features_positive = features_positive[0]
                if isinstance(features_negative, tuple):
                    features_negative = features_negative[0]

                features = torch.cat([features_anchor, features_positive, features_negative], dim=0)

                # Assuming labels for positive and negative pairs are the same as the anchor
                labels = torch.cat([labels, labels, labels], dim=0)
                loss = loss_func(features, labels)
                inputs = img_anchor

            elif args.loss_function =='center_loss':
                
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                logits, features = model(inputs)

                # Calculate top-5 accuracy
                #topk = 5
                #_, topk_predictions = logits.topk(k=5, dim=1)
                #topk_predictions = topk_predictions.t()
                #correct_topk_predictions += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                loss, ce_loss, center_loss_value = loss_func(logits, labels, features)
                correct = topk_predictions.eq(labels.view(1, -1).expand_as(topk_predictions))
                correct_predictions += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                test_loss += loss.item() * inputs.size(0)
                ce_test_loss += ce_loss.item() * inputs.size(0)
                center_test_loss += center_loss_value.item() * inputs.size(0)
                #loss, ce_loss, center_loss_value = combined_loss(logits, labels, features)
                #loss = combined_loss(logits, labels, features) Second version 
                
            else:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                logits, features = model(inputs)

                loss = loss_func(features, labels)

                correct = topk_predictions.eq(labels.view(1, -1).expand_as(topk_predictions))
                correct_predictions += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()   
                test_loss += loss.item() * inputs.size(0)
            

            
    avg_test_accuracy = 100.0 * correct_predictions /(dataset_sizes["test"])
    avg_test_loss = test_loss / dataset_sizes["test"]
    avg_test_ce_loss = ce_test_loss / dataset_sizes["test"]
    avg_test_center_loss = center_test_loss / dataset_sizes["test"]
    
    #avg_test_accuracy_topk = 100.0 * correct_topk_predictions / dataset_sizes["test"]
    logger.info(f'Test Accuracy (Top1): {avg_test_accuracy:.2f}%')
    logger.info(f'Test Loss: {avg_test_loss:.2f}%')
    logger.info(f'Test Cross-Entropy Loss: {avg_test_ce_loss:.4f}%')
    logger.info(f'Test Center Loss: {avg_test_center_loss:.4f}%')
    
    #logger.info(f'Test Accuracy (Top5): {avg_test_accuracy_topk:.2f}%')

    #return avg_test_accuracy
    #avg_test_accuracy = 100.0 * correct_predictions / dataset_sizes["test"]
    #avg_test_loss = test_loss / dataset_sizes["test"]
    #avg_test_ce_loss = ce_test_loss / dataset_sizes["test"]
    #avg_test_center_loss = center_test_loss / dataset_sizes["test"]
    #avg_test_combined_loss = avg_test_ce_loss + avg_test_center_loss
    #avg_test_accuracy_topk = 100.0 * correct_topk_predictions / dataset_sizes["test"]

    #logger.info(f'Test Loss: {avg_test_loss:.4f}%')
    #logger.info(f'Test Combined Loss: {avg_test_combined_loss:.4f}')
    #logger.info(f'Test Cross-Entropy Loss: {avg_test_ce_loss:.4f}%')
    #logger.info(f'Test Center Loss: {avg_test_center_loss:.4f}%')
    #logger.info(f'Test Accuracy (Top1): {avg_test_accuracy:.4f}%')
    #logger.info(f'Test Accuracy (Top5): {avg_test_accuracy_topk:.4f}%')

    if wandb:
        wandb.log({
            #"Test Combined Loss": avg_test_combined_loss,
            "Test Cross-Entropy Loss": avg_test_ce_loss,
            "Test Center Loss": avg_test_center_loss,
            "Test Loss": avg_test_loss, 
            "Test Accuracy (Top1)": avg_test_accuracy,
            #"Test Accuracy (Top5)": avg_test_accuracy_topk,
        })

    return avg_test_accuracy

def save_network_weights(model: nn.Module, weights_name: str) -> None:
    """
    Save the weights of a neural network model to a file.

    Args:
        model (nn.Module): The neural network model.
        weights_name (str): The name of the weights file.
    """
    try:
         # Ensure the output directory exists
        if not os.path.exists(outweights_dir):
            os.makedirs(outweights_dir)
        
        # Construct the full path for the weights file
        weights_path = os.path.join(outweights_dir, weights_name)

        state_dict = model.state_dict()
        torch.save(state_dict, weights_path )
        logger.info(f"Model weights saved to {weights_path}")
    except Exception as e:
        logger.error(f"Failed to save model weights: {e}")



if __name__ == '__main__':

    parser = get_training_config_parser()
    args = parser.parse_args()
    start = time.time()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()

    dataset_loader, dataset_sizes = load_dataloader(args)

    #for batch in dataset_loader["train"]:
        
        #anchor, positive, negative = batch
        #print(f"Anchor batch shape: {anchor.shape}, Positive batch shape: {positive.shape}, Negative batch shape: {negative.shape}")
        #break

    # Check if the validation dataset loader is correctly defined
    if "valid" not in dataset_loader:
        raise ValueError("Validation dataset loader is not defined.")

    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB__SERVICE_WAIT']='120'
    # Initialize Weights and Biases
    wandb.init(
        project="Modeling the Face Recognition System in the Brain",
        config={
            "architecture": args.model,
            "dataset": args.dataset,
            "epochs": args.num_epochs,
            "num_classes": args.num_classes,
        },
    )

    # Choose the model class based on args.model
    model_cls = { "cornet_s": cornet_s, "resnet50": resnet50,
                 "mobilenet": mobilenet_v2,  "vgg16_bn": vgg16_bn, 
                "inception_v3": inception_v3, "FaceNet": FaceNet, 
                "SphereFace": SphereFace}[args.model]

    # Initialize the model
    model = model_cls(args.pretrained, args.num_classes, args.n_input_channels, args.transfer, args.in_weights)
    model.to(device)
    init_weights_randomly(model)
    

    loss_func = get_loss_function(args,device)
    

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

    experiment_name = f"final_{args.model}_{args.num_classes}_{args.seed}"
    
        
    model_ft = train_model(model,optimizer_ft, exp_lr_scheduler, args.num_epochs, loss_func,  dataset_loader, dataset_sizes, wandb)
    acc = test_model(model_ft, dataset_loader, dataset_sizes,loss_func)
        
    wandb.log({"Test Acc": acc})

    logger.info("Training complete.")

    time_training = time.time() - start
    logger.info('Training ended in %s h %s m %s s' % (time_training // 3600, (time_training % 3600) // 60, time_training % 60))

    # Save weights after training
    #save_network_weights(model_ft, f"{args.out_weights}_{args.seed}.pth")
    if args.out_weights:
        out_weights_path = os.path.join('outweights_dir', f"{args.out_weights}_{args.seed}.pth")
        save_network_weights(model_ft, out_weights_path)
