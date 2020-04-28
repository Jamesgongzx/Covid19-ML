import torch.nn as nn
import torch
from torch import optim
from torch.optim import lr_scheduler
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Code derived from https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
# ConvNet as fixed feature extractor
def resnet(layers=50, learning_rate=0.01, momentum=0.9):
    if layers == 18:
        model_conv = models.resnet18(pretrained=True)
    elif layers == 101:
        model_conv = models.resnet101(pretrained=True)
    elif layers == 152:
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
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=momentum)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    return model_conv, criterion, optimizer_conv, exp_lr_scheduler

# Code derived from https://github.com/jrzech/reproduce-chexnet/blob/master/model.py
def chexnet(learning_rate=0.01, momentum=0.9, weight_decay=1e-4):
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    N_LABELS = 2
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, N_LABELS),
        nn.Sigmoid())
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay)
