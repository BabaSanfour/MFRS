import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
path_weights = "/home/hamza97/scratch/net_weights/"

def train_network(name, model, criterion, optimizer, scheduler, num_epochs, dataset_loader, dataset_sizes):

    """Train the model using the train and validation datasets"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = -1.0
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

#    fig_path = path_plot+'%s.png'%(name)
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

#    ax1.plot(list_trainLoss, "--", label='train loss')
#    ax2.plot(list_trainAcc,"--", label='train accuracy')
#    ax1.plot(list_valLoss, label='valid loss')
#    ax2.plot(list_valAcc, label='valid accuracy')
#    ax2.legend()
#    ax1.legend()
#    fig.savefig(fig_path)
#    fig.show()


    print()
    time_elapsed = time.time() - since
    print('Training complete in %s h %s m %s s' % (time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
    print('Best val Acc: %.2f%%' % (best_acc))

    return best_model

def test_network(model, dataset_loader, dataset_sizes):

    # torch works with gpu
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_dev_correct=0
    tot_correct_topk=0
    test_loss = 0.0
    # The loss function
    criterion = nn.CrossEntropyLoss()

    for inputs_, labels_ in dataset_loader['test']:
        # To work with GPU
        inputs = inputs_.to(device)
        labels = labels_.to(device)
        outputs = model(inputs)
        # top 5 accuracy
        topk = 5
        _, y_pred = outputs.topk(k=5, dim=1)
        y_pred = y_pred.t()
        labels_reshaped = labels.view(1, -1).expand_as(y_pred)
        correct = (y_pred == labels_reshaped)
        flattened_indicator_which_topk_matched_truth = correct.reshape(-1).float()
        tot_correct_topk += flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)

        n_dev_correct += (torch.max(outputs, 1)[1].view(labels.data.size()) == labels.data).sum().item()

        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

    avg_test_acc = 100. * n_dev_correct/dataset_sizes["test"]
    avg_test_loss = test_loss / dataset_sizes["test"]
    avg_test_acc5 = 100. * tot_correct_topk/dataset_sizes["test"]

    print('Test Loss: %.2f' % (avg_test_loss))
    print('Test Accuracy (Top1): %.2f%%' % (avg_test_acc))
    print('Test Accuracy (Top5): %.2f%%' % (avg_test_acc5))


    return avg_test_acc

def save_network_weights(model_ft,  file) :
    """Save the network after training"""
    state = model_ft.state_dict()
    torch.save(state,path_weights+file)
