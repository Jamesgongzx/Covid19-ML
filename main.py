import csv
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import pickle
import argparse
import models

plt.ion()  # interactive mode

# Parse command line arguments
parser = argparse.ArgumentParser(description='COVID-19 Classification')
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--model', type=str, default="resnet50")
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


def train_resnet(model, criterion, optimizer, scheduler, epochs):
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


def train_chexnet(model, criterion, optimizer, epochs, learning_rate, weight_decay):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
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

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                print("decay loss from " + str(learning_rate) + " to " +
                      str(learning_rate / 10) + " as not seeing improvement in val loss")
                learning_rate = learning_rate / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=learning_rate,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(learning_rate))

            # Check if model has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

            # log training and validation loss over each epoch
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, last_train_loss, epoch_loss))

        # break if no val loss improvement in 3 epochs
        if (epoch - best_epoch) >= 3:
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
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
    predictions = predictions.astype(int)
    print('Test Dataset Predictions: ' + str(predictions))
    file_path = os.path.join("predictions",
                             args.model + "_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + ".csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["Id", "Predicted"])
        for i in range(len(predictions)):
            result = "True" if predictions[i] else "False"
            writer.writerow([i, result])


def main():
    if "resnet" in args.model:
        if args.model == "resnet18":
            model, model_criterion, model_optimizer, model_scheduler = models.resnet(18, args.learning_rate,
                                                                                     args.momentum)
        elif args.model == "resnet101":
            model, model_criterion, model_optimizer, model_scheduler = models.resnet(101, args.learning_rate,
                                                                                     args.momentum)
        elif args.model == "resnet152":
            model, model_criterion, model_optimizer, model_scheduler = models.resnet(152, args.learning_rate,
                                                                                     args.momentum)
        else:
            print("Using default resnet model: resnet50")
            model, model_criterion, model_optimizer, model_scheduler = models.resnet(50, args.learning_rate,
                                                                                     args.momentum)
        model = train_resnet(model, model_criterion, model_optimizer, model_scheduler, epochs=args.epochs)
        predict_model(model)
    elif args.model == "chexnet":
        model, model_criterion, model_optimizer = models.chexnet(args.learning_rate, args.momentum, args.weight_decay)
        model = train_chexnet(model, model_criterion, model_optimizer, epochs=args.epochs,
                              learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        predict_model(model)
    else:
        print("Please input a valid model")


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

    main()
