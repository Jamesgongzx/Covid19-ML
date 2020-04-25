import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import pickle
import argparse

plt.ion()  # interactive mode

# Parse command line arguments
parser = argparse.ArgumentParser(description='COVID-19 Classification')
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--layers', type=float, default=50)
args = parser.parse_args()


class CovidDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


class CovidDatasetTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx]


def make_data_loaders():
    train_dataset_size = len(train_labels)
    split = int(train_dataset_size * 0.2)

    indices = list(range(train_dataset_size))
    # This seed gives four 0 labels in the validation_dataset
    np.random.seed(6)
    np.random.shuffle(indices)

    train_dataset = CovidDataset(train_imgs[indices[split:]], train_labels[indices[split:]])
    validation_dataset = CovidDataset(train_imgs[indices[:split]], train_labels[indices[:split]])
    test_dataset = CovidDatasetTest(test_imgs)

    print("Training labels: " + str(train_dataset.labels.cpu().numpy()))
    print("Validation labels: " + str(validation_dataset.labels.cpu().numpy()))

    return {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "test": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }


# ConvNet as fixed feature extractor
def create_resnet():
    if args.layers == 18:
        model_conv = models.resnet18(pretrained=True)
    elif args.layers == 101:
        model_conv = models.resnet101(pretrained=True)
    elif args.layers == 152:
        model_conv = models.resnet152(pretrained=True)
    else:
        model_conv = models.resnet50(pretrained=True)

    # Freeze all the network except the final layer.
    # Freeze the parameters so that the gradients are not computed in backward().
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    # Only parameters of final layer are being optimized
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=args.learning_rate, momentum=args.momentum)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    return model_conv, criterion, optimizer_conv, exp_lr_scheduler


def train_model(model, criterion, optimizer, scheduler, epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_loss = float("inf")
    best_acc_train_acc = 0.0
    best_acc_train_loss = float("inf")

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        train_acc = 0.0
        train_loss = 0.0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss < best_acc_loss):
                    best_acc = epoch_acc
                    best_acc_loss = epoch_loss
                    best_acc_train_acc = train_acc
                    best_acc_train_loss = train_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {:4f}. Validation Loss: {:4f} '.format(best_acc, best_acc_loss))
    print(
        'Corresponding Training Accuracy: {:4f}. Training Loss: {:4f} '.format(best_acc_train_acc, best_acc_train_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def predict_model(model):
    was_training = model.training
    model.eval()
    predictions = np.array([])
    with torch.no_grad():
        for i, (inputs) in enumerate(data_loaders['test']):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions = np.concatenate([predictions, preds.cpu().numpy()])
        model.train(mode=was_training)
    print('Test Dataset Predictions: ' + str(predictions))


# Code derived from https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
if __name__ == '__main__':
    train_imgs = pickle.load(open("data/train_images_512.pk", 'rb'), encoding='bytes')
    train_labels = pickle.load(open("data/train_labels_512.pk", 'rb'), encoding='bytes')
    test_imgs = pickle.load(open("data/test_images_512.pk", 'rb'), encoding='bytes')

    data_loaders = make_data_loaders()
    dataset_sizes = {'train': int(len(data_loaders['train'].dataset)),
                     'val': int(len(data_loaders['val'].dataset)),
                     'test': len(data_loaders['test'].dataset)}
    print(dataset_sizes)

    class_names = ['covid', 'background']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resNet, model_criterion, model_optimizer, model_scheduler = create_resnet()
    resNet = train_model(resNet, model_criterion, model_optimizer, model_scheduler, epochs=args.epochs)
    predict_model(resNet)
