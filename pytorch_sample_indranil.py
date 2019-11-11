import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import pdb
import models
from pytorch_mvm_class import *
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES']='0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

## To Indranil & Mustafa: This is for using 'for loops' in mvm_tensor. Just execute with '-i' at command line
# ind = False
# for i in range(len(sys.argv)):
#     if sys.argv[i] == '-i':
#         ind = True

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', default=100, type=int,
                     metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-i', default=False,
                     metavar='N', help='turn on Ind feature')
parser.add_argument('--arch', action='store', default='vgg',
            help='the architecture for the network: resnet')
parser.add_argument('--model', '-a', metavar='MODEL', default='vgg',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet)')
args = parser.parse_args()
if args.i == 'True':
    ind = True
else:
    ind = False
#

inputs = torch.tensor([[[[-1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]], [[[1.,0,1],[-2,1,0],[1,-2,1]],[[2,-3,1],[-2,0,-1],[4,-2,-1]],[[-3,2,1],[0,2,1],[-5,3,2]]]])/10
#inputs = torch.tensor([[[[1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]], [[[-1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]]])

labels = torch.tensor([1, 1])
weights = torch.tensor([[[[-2.,1],[-1,2]],[[-4,2],[0,1]],[[-1,0],[-3,-2]]],[[[2.,1],[1,2]],[[3,2],[1,1]],[[1,2],[3,2]]]])/10
trainloader = [[inputs, labels]]
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform =transforms.Compose([transforms.ToTensor()]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
inputs, labels = next(iter(trainloader))


print('==> building model',args.arch,'...')
if args.arch == 'vgg':
    #print(models.__dict__)
    model = models.__dict__[args.model]
    #model_config = {'input_size': args.input_size, 'dataset': args.dataset}
    print(model)
else:
    raise Exception(args.arch+' is currently not supported')


model = model()
model_mvm = models.__dict__['vgg_mvm']
model_mvm = model_mvm(ind)
#pdb.set_trace()

print('==> Initializing model parameters ...')
weights_conv = []
for m in model.modules():
    
  #  print (m)for m in model.modules():
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        weights_conv.append(m.weight.data.clone())
    #print(m.weight.data)
    #raw_input()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.data.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            weights_lin = m.weight.data.clone()
            if m.bias is not None:
               m.bias.data.uniform_(-stdv, stdv)

i=0
for m in model_mvm.modules():
    
  #  print (m)for m in model.modules():
    if isinstance(m, nn.Conv2d):
        m.weight.data = weights_conv[i]
        i = i+1
    #print(m.weight.data)
    #raw_input()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
            m.weight.data = weights_lin
model.cuda()
model_mvm.cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)
# mynet.to(device)
inputs = inputs.to(device)

result_net = model(inputs)
print(result_net[0][0])#[0,:2])

result_mynet = model_mvm(inputs)
print(result_mynet[0][0])#[0,:2])
#print(torch.norm(result_net-result_mynet))

