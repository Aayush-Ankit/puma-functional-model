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
os.environ['CUDA_VISIBLE_DEVICES']='3'

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
inputs, labels = next(iter(trainloader))
#inputs = torch.rand(2,32,5,5).mul_(2).sub_(1)
#weights_conv1 = torch.rand(64,3,3,3).sub_(0.5)
#weights_conv2 = torch.rand(64,64,3,3).sub_(0.5)
#
#weights_conv3 = torch.rand(64,64,3,3).sub_(0.5)
#weights_conv4 = torch.rand(64,64,3,3).sub_(0.5)
#
#weights_conv5 = torch.rand(256,128,3,3).sub_(0.5)
#weights_conv6 = torch.rand(256,256,3,3).sub_(0.5)
#
#weights_conv7 = torch.rand(512,256,3,3).sub_(0.5)
#weights_conv6= torch.rand(512,512,3,3).sub_(0.5)
#
#weights_conv9 = torch.rand(512,512,3,3).sub_(0.5)
#weights_conv10 = torch.rand(512,512,3,3).sub_(0.5)
#
#weights_lin = torch.rand(10,512).sub_(0.5).mul_(0.5)

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

#pdb.set_trace()
# weights_conv1 = torch.rand(64,3,3,3).sub_(0.5)

# weights_conv2 = torch.rand(64,64,3,3).sub_(0.5)
# weights_conv3 = torch.rand(64,64,3,3).sub_(0.5)
# weights_conv4 = torch.rand(64,64,3,3).sub_(0.5)
# weights_conv5 = torch.rand(64,64,3,3).sub_(0.5)


# weights_conv6 = torch.rand(128,64,3,3).sub_(0.5)
# weights_conv7 = torch.rand(128,128,3,3).sub_(0.5)
# weights_conv8 = torch.rand(128,128,3,3).sub_(0.5)
# weights_conv9 = torch.rand(128,128,3,3).sub_(0.5)



# weights_conv10 = torch.rand(256,128,3,3).sub_(0.5)
# weights_conv11 = torch.rand(256,256,3,3).sub_(0.5)
# weights_conv12 = torch.rand(256,256,3,3).sub_(0.5)
# weights_conv13 = torch.rand(256,256,3,3).sub_(0.5)


# weights_lin = torch.rand(10,256).sub_(0.5).mul_(0.5)


#inputs_lin = torch.tensor([[-1.,0,1,2,-2],[5, 4, 3, 2, 1]])/10
#weights_lin = torch.tensor([[-1.,0,1,2,-2],[5, 4, 3, 2, 1],[-1, -2, -3, -4, -5]])/10


# class my_Net(nn.Module):
#     def __init__(self):
#         super(my_Net, self).__init__()
        
#         self.conv1 = Conv2d_mvm(3,64,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv1.weight.data = torch.clone(weights_conv[0])
    
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn1.bias.data.zero_()
#         self.bn1.weight.data.fill_(1)


#         self.conv2 = Conv2d_mvm(64,64,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv2.weight.data = torch.clone(weights_conv[1])
#         self.maxpool1 = nn.MaxPool2d(2,2)

#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn2.bias.data.zero_() 
#         self.bn2.weight.data.fill_(1)


#         self.conv3 = Conv2d_mvm(64,64,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv3.weight.data = torch.clone(weights_conv[2])

#         self.bn3 = nn.BatchNorm2d(64)
#         self.bn3.bias.data.zero_() 
#         self.bn3.weight.data.fill_(1)


#         self.conv4 = Conv2d_mvm(64,64,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv4.weight.data = torch.clone(weights_conv[3])

#         self.bn4 = nn.BatchNorm2d(64)
#         self.bn4.bias.data.zero_() 
#         self.bn4.weight.data.fill_(1)

#         self.maxpool2 = nn.MaxPool2d(2,2)

#         self.conv5 = Conv2d_mvm(64,64,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv5.weight.data = torch.clone(weights_conv[4])

#         self.bn5 = nn.BatchNorm2d(128)
#         self.bn5.bias.data.zero_() 
#         self.bn5.weight.data.fill_(1)


#         self.conv6= Conv2d_mvm(64,128,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv6.weight.data = torch.clone(weights_conv[5])

#         self.bn6= nn.BatchNorm2d(128)
#         self.bn6.bias.data.zero_() 
#         self.bn6.weight.data.fill_(1)


#         self.conv7 = Conv2d_mvm(128,128,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv7.weight.data = torch.clone(weights_conv[6])
        
#         self.bn7 = nn.BatchNorm2d(128)
#         self.bn7.bias.data.zero_() 
#         self.bn7.weight.data.fill_(1)
        
#         self.maxpool3 = nn.MaxPool2d(2,2)
        
        
#         self.conv8 = Conv2d_mvm(128,128,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv8.weight.data = torch.clone(weights_conv[7])
        
#         self.bn8 = nn.BatchNorm2d(128)
#         self.bn8.bias.data.zero_() 
#         self.bn8.weight.data.fill_(1)
        
#         self.conv9 = Conv2d_mvm(128,128,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv9.weight.data = torch.clone(weights_conv[8])
        
#         self.bn9 = nn.BatchNorm2d(128)
#         self.bn9.bias.data.zero_() 
#         self.bn9.weight.data.fill_(1)
        
#         self.maxpool4 = nn.MaxPool2d(2,2)
        
#         self.conv10 = Conv2d_mvm(128,256,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv10.weight.data = torch.clone(weights_conv[9])
        
#         self.bn10 = nn.BatchNorm2d(256)
#         self.bn10.bias.data.zero_() 
#         self.bn10.weight.data.fill_(1)

#         self.conv11 = Conv2d_mvm(256,256,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv11.weight.data = torch.clone(weights_conv[10])
        
#         self.bn11 = nn.BatchNorm2d(256)
#         self.bn11.bias.data.zero_() 
#         self.bn11.weight.data.fill_(1)

#         self.conv12 = Conv2d_mvm(256,256,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv12.weight.data = torch.clone(weights_conv[11])
        
#         self.bn12 = nn.BatchNorm2d(256)
#         self.bn12.bias.data.zero_() 
#         self.bn12.weight.data.fill_(1)

#         self.conv13 = Conv2d_mvm(256,256,3, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.conv13.weight.data = torch.clone(weights_conv[12])
        
#         self.bn13 = nn.BatchNorm2d(256)
#         self.bn13.bias.data.zero_() 
#         self.bn13.weight.data.fill_(1)

#         self.maxpool5 = nn.MaxPool2d(2,2)
        
        
#         self.avgpool = nn.AvgPool2d(6)
        
#         self.linear = Linear_mvm(256,10, bit_slice = 4, bit_stream = 4, bias=False, ind=ind)
#         self.linear.weight.data = torch.clone(weights_lin)

#     def forward(self, x):
#         t = time.time()
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.bn1(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.bn2(x)
#         #x = self.maxpool1(x)
        
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.bn3(x)
#         x = self.conv4(x)
#         x = F.relu(x)
#         x = self.bn4(x)
#         #x = self.maxpool2(x)

#         x = self.conv5(x)
#         x = F.relu(x)
#         x = self.bn5(x)
#         x = self.conv6(x)
#         x = F.relu(x)
#         x = self.bn6(x)
#         #x = self.maxpool3(x)
        
#         x = self.conv7(x)
#         x = F.relu(x)
#         x = self.bn7(x)
#         x = self.conv8(x)
#         x = F.relu(x)
#         x = self.bn8(x)
#         #x = self.maxpool4(x)
        
        
#         x = self.conv9(x)
#         x = F.relu(x)
#         x = self.bn9(x)
#         x = self.conv10(x)
#         x = F.relu(x)
#         x = self.bn10(x)
#         #x = self.maxpool5(x)
        
#         x = self.conv11(x)
#         x = F.relu(x)
#         x = self.bn11(x)
#         x = self.conv12(x)
#         x = F.relu(x)
#         x = self.bn12(x)
#         x = self.conv13(x)
#         x = F.relu(x)
#         x = self.bn13(x)
#         #x = self.maxpool5(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0),-1)
#         x = self.linear(x)
#         t1 = time.time()
#         print('Time taken: ',t1-t)
#         return x


#net = Net()
# mynet = my_Net()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)
# mynet.to(device)
inputs = inputs.to(device)

result_net = model(inputs)
print(result_net[0][0])#[0,:2])

result_mynet = model_mvm(inputs)
print(result_mynet[0][0])#[0,:2])
#print(torch.norm(result_net-result_mynet))

