import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms

import argparse
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from models.resnet import *
from models.mvcnn import *
import util
from logger import Logger
from custom_dataset_our_case import MultiViewDataSet

import globals


def train_MVCNN(case_description):
    print("\nTrain MVCNN\n")
    MVCNN = 'mvcnn'
    RESNET = 'resnet'
    MODELS = [RESNET, MVCNN]

    DATA_PATH = globals.DATA_PATH
    DEPTH = None
    MODEL = MODELS[1]
    EPOCHS = 100
    BATCH_SIZE = 10
    LR = 0.0001
    MOMENTUM = 0.9
    LR_DECAY_FREQ = 30
    LR_DECAY = 0.1
    PRINT_FREQ = 10
    RESUME = ""
    PRETRAINED = True
    
    REMOTE = globals.REMOTE
    case_description = case_description


    print('Loading data')

    transform = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dset_train = MultiViewDataSet(DATA_PATH, 'train', transform=transform)
    train_loader = DataLoader(dset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    ## Got rid of this, not using validation here.
    # dset_val = MultiViewDataSet(DATA_PATH, 'test', transform=transform)
    # val_loader = DataLoader(dset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    classes = dset_train.classes
    print(len(classes), classes)

    if MODEL == RESNET:
        if DEPTH == 18:
            model = resnet18(pretrained=PRETRAINED, num_classes=len(classes))
        elif DEPTH == 34:
            model = resnet34(pretrained=PRETRAINED, num_classes=len(classes))
        elif DEPTH == 50:
            model = resnet50(pretrained=PRETRAINED, num_classes=len(classes))
        elif DEPTH == 101:
            model = resnet101(pretrained=PRETRAINED, num_classes=len(classes))
        elif DEPTH == 152:
            model = resnet152(pretrained=PRETRAINED, num_classes=len(classes))
        else:
            raise Exception('Specify number of layers for resnet in command line. --resnet N')
        print('Using ' + MODEL + str(DEPTH))
    else:
        # number of ModelNet40 needs to match loaded pre-trained model
        model = mvcnn(pretrained=PRETRAINED, num_classes=40)
        print('Using ' + MODEL)

    
    cudnn.benchmark = True

    print('Running on ' + str(device))

    """
    Load pre-trained model and freeze weights for training.
    This is done by setting param.requires_grad to False
    """

    """Just added this check to load my pretrained model instead of copying it to the repo and having a duplicate"""
    if REMOTE:
        PATH = "../../MVCNN_Peter/checkpoint/mvcnn18_checkpoint.pth.tar"
    else:
        PATH = "checkpoint/model_from_pete.tar"

    loaded_model = torch.load(PATH)
    model.load_state_dict(loaded_model['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, len(classes))

    model.to(device)

    print(model)

    logger = Logger('logs')

    # Loss and Optimizer
    lr = LR
    n_epochs = EPOCHS
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    start_epoch = 0


    # Helper functions
    def load_checkpoint():
        global best_acc, start_epoch
        # Load checkpoint.
        print('\n==> Loading checkpoint..')
        assert os.path.isfile(RESUME), 'Error: no checkpoint file found!'

        checkpoint = torch.load(RESUME)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    def train():
        train_size = len(train_loader)
        loss = None
        total = 0
        correct = 0
        for i, (inputs, targets) in enumerate(train_loader):

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
            correct += (predicted.cpu() == targets.cpu()).sum().item()

            """
            print("total: ", total)
            print("correct: ", correct)
            print()
            """

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % PRINT_FREQ == 0:
                print("\tIter [%d/%d] Loss: %.4f" % (i + 1, train_size, loss.item()))

        return loss, int(float(float(correct)/float(total))*100)

    # Training / Eval loop
    if RESUME:
        load_checkpoint()

    best_acc = 0
    best_loss = 0
    loss_values = []
    acc_values = []
    for epoch in range(start_epoch, n_epochs):
        print('\n-----------------------------------')
        print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
        start = time.time()

        model.train()
        (t_loss, t_acc) = train()
        loss_values.append(t_loss)
        acc_values.append(t_acc)

        print("Total loss: " + str(t_loss))
        print("Accuracy: " + str(t_acc) + "%")

        print('Time taken: %.2f sec.' % (time.time() - start))

        if t_acc > best_acc:
            print("UPDATE")
            print("UPDATE")
            print("UPDATE")
            print("UPDATE")
            print("UPDATE")
            best_acc = t_acc
            best_loss = t_loss
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss_per_epoch': loss_values,
                'acc_per_epoch': acc_values,
                'optimizer': optimizer.state_dict(),
            }, MODEL, DEPTH, case_description)

        # Decaying Learning Rate
        if (epoch + 1) % LR_DECAY_FREQ == 0:
            lr *= LR_DECAY
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print('Learning rate:', lr)

    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(loss_values, 'r')
    axs[1].plot(acc_values, 'b')

    if not REMOTE:
        plt.show()
    else:
        plt.savefig("plots/training.png")

