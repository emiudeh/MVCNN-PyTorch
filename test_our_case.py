import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms

import argparse
import numpy as np

import matplotlib.pyplot as plt
import time
import os

from models.resnet import *
from models.mvcnn import *
import util
from logger import Logger
from custom_dataset_our_case import MultiViewDataSet

def test_MVCNN(test_description):
    MVCNN = 'mvcnn'
    RESNET = 'resnet'
    MODELS = [RESNET,MVCNN]
    DATA_PATH = 'views_our_case/classes'
    DEPTH = 18
    MODEL = MODELS[1]
    PRETRAINED = True
    test_description = test_description

    criterion = nn.CrossEntropyLoss()

    print('Loading data')

    transform = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dset_test = MultiViewDataSet(DATA_PATH, 'test', transform=transform)
    test_loader = DataLoader(dset_test, batch_size=4, shuffle=False, num_workers=2)


    if MODEL == RESNET:
        if DEPTH == 18:
            model = resnet18(pretrained=PRETRAINED, num_classes=4)
        elif DEPTH == 34:
            model = resnet34(pretrained=PRETRAINED, num_classes=4)
        elif DEPTH == 50:
            model = resnet50(pretrained=PRETRAINED, num_classes=4)
        elif DEPTH == 101:
            model = resnet101(pretrained=PRETRAINED, num_classes=4)
        elif DEPTH == 152:
            model = resnet152(pretrained=PRETRAINED, num_classes=4)
        else:
            raise Exception('Specify number of layers for resnet in command line. --resnet N')
        print('Using ' + MODEL + str(DEPTH))
    else:
        model = mvcnn(pretrained=PRETRAINED, num_classes=4)
        print('Using ' + MODEL)


    model.to(device)
    cudnn.benchmark = True

    print('Running on ' + str(device))




    # The above code mostly sets up stuff. Now is the important logic
    ###########
    PATH = "checkpoint/5_instances/mvcnn_checkpoint.pth.tar"
    loaded_model = torch.load(PATH)
    model.load_state_dict(loaded_model['state_dict'])
    model.eval()

    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(loaded_model['loss_per_epoch'], 'r')
    axs[1].plot(loaded_model['acc_per_epoch'], 'b')
    plt.show()

    correct = 0
    total = 0
    print("we have total of ", len(test_loader), "batches")
    for i, (inputs, targets) in enumerate(test_loader):
        print("..processing batch", i)
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    acc = 100 * correct.item() / total
    print("total Accuracy:", acc)

    output_file = PATH[:PATH.rfind('/') + 1] + "output.txt"
    f = open(output_file, "a+")
    f.write(test_description)
    f.write("\nAccuracy: %d\rLoss: %d\r\n\n" % (acc, loss))
    print(output_file)

    ###########