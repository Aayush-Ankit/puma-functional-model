import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import torch.backends.cudnn as cudnn
import numpy as np
import random
from src.pytorch_mvm_class_v3 import *
import pdb

manual_seed=0
torch.manual_seed(manual_seed) 
torch.cuda.manual_seed_all(manual_seed) 
np.random.seed(manual_seed) 
random.seed(manual_seed) 
os.environ['PYTHONHASHSEED'] = str(manual_seed) 
cudnn.deterministic = True 
cudnn.benchmark = False 

os.environ['CUDA_VISIBLE_DEVICES']= '1'

## To Indranil & Mustafa: This is for using 'for loops' in mvm_tensor. Just execute with '-i' at command line
ind = False
for i in range(len(sys.argv)):
    if sys.argv[i] == '-i':
        ind = True

inputs = torch.tensor([[[[-1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]], [[[1.,0,1],[-2,1,0],[1,-2,1]],[[2,-3,1],[-2,0,-1],[4,-2,-1]],[[-3,2,1],[0,2,1],[-5,3,2]]]])/10
#inputs = torch.tensor([[[[1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]], [[[-1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]]])

labels = torch.tensor([1, 1])
weights = torch.tensor([[[[-2.,1],[-1,2]],[[-4,2],[0,1]],[[-1,0],[-3,-2]]],[[[2.,1],[1,2]],[[3,2],[1,1]],[[1,2],[3,2]]]])/10
trainloader = [[inputs, labels]]
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform =transforms.Compose([transforms.ToTensor()]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
inputs, labels = next(iter(trainloader))
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform =transforms.Compose([transforms.ToTensor()]))
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)


#inputs = torch.rand(64,64,32,32).sub(0.5).mul(0.5)
#weights = torch.rand(64,64,3,3).sub(0.5).mul(0.5)


#weights_lin = torch.rand(10,288).sub_(0.5).mul_(0.5)


#inputs_lin = torch.tensor([[-1.,0,1,2,-2],[5, 4, 3, 2, 1]])/10
#weights_lin = torch.tensor([[-1.,0,1,2,-2],[5, 4, 3, 2, 1],[-1, -2, -3, -4, -5]])/10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3,512,3, bias=False, stride =1, padding=1)
        self.conv2 = nn.Conv2d(512,512,3, bias=False, stride =2, padding=1)
        self.conv3 = nn.Conv2d(512,512,3, bias=False, stride =2, padding=1)
        self.avgpool = nn.AvgPool2d(6)
        self.linear = nn.Linear(512,10, bias = False)
        #print(self.linear.weight.data.shape)
        #self.linear.weight.data = torch.clone(weights_lin)

    def forward(self, x):
        #self.conv1.weight.data = torch.clone(weights_conv[0])
        print(self.conv1.weight.data[0][0][0])
        x = self.conv1(x)
        y = x.clone()

        #self.conv2.weight.data = torch.clone(weights_conv[1])
        print(self.conv2.weight.data[0][0][0])
        x = self.conv2(x)
        z = x.clone()
        print(self.conv3.weight.data[0][0][0])
        x = self.conv3(x)
        w = x.clone()
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x, y, z, w


class my_Net(nn.Module):
    def __init__(self):
        super(my_Net, self).__init__()

        self.conv1 = nn.Conv2d(3,512,3,  stride=1, padding=1, bias=False)# bit_slice=4, bit_stream=4, weight_bits=16, weight_bit_frac=14, input_bits=16, input_bit_frac=14, adc_bit=14, acm_bits=32, acm_bit_frac=24, ind=ind, loop=True)   # --> my custom module for mvm
        self.conv2 = Conv2d_mvm(512,512,3, stride=2, padding=1, bias=False, bit_slice=4, bit_stream=4, weight_bits=16, weight_bit_frac=14, input_bits=16, input_bit_frac=14, adc_bit=14, acm_bits=32, acm_bit_frac=24, ind=ind, loop=False)
        self.conv3 = Conv2d_mvm(512,512,3, stride=2, padding=1, bias=False, bit_slice=4, bit_stream=4, weight_bits=16, weight_bit_frac=14, input_bits=16, input_bit_frac=14, adc_bit=14, acm_bits=32, acm_bit_frac=24, ind=ind, loop=False)
        self.avgpool = nn.AvgPool2d(6)
        self.linear = Linear_mvm(512,10, bias=False, bit_slice = 4, bit_stream = 4, weight_bits=16, weight_bit_frac=14, input_bits=16, input_bit_frac=14, adc_bit=14, acm_bits=32, acm_bit_frac=24, ind = ind)

        #self.linear.weight.data = torch.clone(weights_lin)

    def forward(self, x):
        self.conv1.weight.data = torch.clone(weights_conv[0])
        print(self.conv1.weight.data[0][0][0])
        x = self.conv1(x)
        y = x.clone()

        self.conv2.weight.data = torch.clone(weights_conv[1])
        print(self.conv2.weight.data[0][0][0])
        print(self.conv2.ind)
        x = self.conv2(x)
        z = x.clone()
        
        self.conv3.weight.data = torch.clone(weights_conv[2])
        print(self.conv3.weight.data[0][0][0])
        x = self.conv3(x)
        w = x.clone()
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        self.linear.weight.data = torch.clone(weights_lin)
        x = self.linear(x)
        return x, y, z, w


net = Net()
mynet = my_Net()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
mynet.to(device)
inputs = inputs.to(device)
weights_conv = []
for m in net.modules():
    
  #  print (m)for m in model.modules():
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        weights_conv.append(m.weight.data.clone())
    elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.data.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            weights_lin = m.weight.data.clone()
            if m.bias is not None:
               m.bias.data.uniform_(-stdv, stdv)


time_net=[]
time_mynet=[]
for i in range(3):
    begin = time.time()
    print('net:')
    result_net, conv1_out,conv2_out, conv3_out = net(inputs)
    
    end = time.time()
    time_net.append(end-begin)
    print('mynet:')
    result_mynet, conv1_out_mvm, conv2_out_mvm, conv3_out_mvm = mynet(inputs)
    
    end2 = time.time()
    time_mynet.append(end2-end)
    torch.cuda.empty_cache()
    
print('net average: ',sum(time_net)/len(time_net))

print('mynet average: ',sum(time_mynet)/len(time_mynet))

#begin = time.time()
#result_net, conv1_out,conv2_out, conv3_out = net(inputs)
#
#end = time.time()
#print('time for net:',end-begin)
#
#result_mynet, conv1_out_mvm, conv2_out_mvm, conv3_out_mvm = mynet(inputs)
#
#end2 = time.time()
#print('time for mynet:',end2-end)


print('result_net:',result_net[0])
print('conv1_out:',conv1_out[0][0][0])

print('result_mynet:',result_mynet[0])

print('conv1_out_mvm:',conv1_out_mvm[0][0][0])
print('result_net norm:',torch.norm(result_net-result_mynet))
print('Conv norm', torch.norm(conv1_out-conv1_out_mvm))
