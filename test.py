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
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--lr-decay-freq', default=30, type=float,
                    metavar='W', help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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
test_loader = DataLoader(dset_test, batch_size=1, shuffle=False, num_workers=2)


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


###########
PATH = "checkpoint/resnet18_checkpoint.pth.tar"
cp = torch.load(PATH)
model.load_state_dict(cp['state_dict'])
model.eval()

correct = 0
for i, (inputs, targets) in enumerate(test_loader):
    print("ANOTHER DATA =====================")
    with torch.no_grad():
        # Convert from list of 3D to 4D
        inputs = np.stack(inputs, axis=1)

        inputs = torch.from_numpy(inputs)

        inputs, targets = inputs.cuda(device), targets.cuda(device)
        inputs, targets = Variable(inputs), Variable(targets)

        # compute output
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        if (predicted.cpu() == targets.cpu()):
            correct = correct + 1

        print("*******************")
        print(i)
        # print(outputs.data)
        print("so far:", 100 * correct/(i+1))
        # print(predicted.cpu())
        # print(targets.cpu())
        print("*******************")

###########


