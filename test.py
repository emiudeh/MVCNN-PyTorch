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

from models.resnet import *
from models.mvcnn import *
import util
from logger import Logger
from custom_dataset import MultiViewDataSet

MVCNN = 'mvcnn'
RESNET = 'resnet'
MODELS = [RESNET,MVCNN]

parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152], type=int, metavar='N', default=18, help='resnet depth (default: resnet18)')
parser.add_argument('--model', '-m', metavar='MODEL', default=RESNET, choices=MODELS,
                    help='pretrained model: ' + ' | '.join(MODELS) + ' (default: {})'.format(RESNET))
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

args = parser.parse_args()

print('Loading data')

transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize(224),
    transforms.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
dset_test = MultiViewDataSet(args.data, 'test', transform=transform)
test_loader = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=2)


if args.model == RESNET:
    if args.depth == 18:
        model = resnet18(pretrained=args.pretrained, num_classes=40)
    elif args.depth == 34:
        model = resnet34(pretrained=args.pretrained, num_classes=40)
    elif args.depth == 50:
        model = resnet50(pretrained=args.pretrained, num_classes=40)
    elif args.depth == 101:
        model = resnet101(pretrained=args.pretrained, num_classes=40)
    elif args.depth == 152:
        model = resnet152(pretrained=args.pretrained, num_classes=40)
    else:
        raise Exception('Specify number of layers for resnet in command line. --resnet N')
    print('Using ' + args.model + str(args.depth))
else:
    model = mvcnn(pretrained=args.pretrained,num_classes=40)
    print('Using ' + args.model)


model.to(device)
cudnn.benchmark = True

print('Running on ' + str(device))




# The above code mostly sets up stuff. Now is the important logic
###########
PATH = "checkpoint/model_from_pete.tar"
loaded_model = torch.load(PATH)
model.load_state_dict(loaded_model['state_dict'])
model.eval()

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
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += (predicted.cpu() == targets.cpu()).sum()

acc = 100 * correct.item() / total
print("total Accuracy:", acc)

###########

