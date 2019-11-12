import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

from pytorch_mvm_class import *

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
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform =transforms.Compose([transforms.ToTensor()]))
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

inputs = torch.rand(2,16,5,5).sub(0.5).div(10)
weights = torch.rand(32,16,3,3).sub(0.5).div(10)

weights_lin = torch.rand(10,288).sub_(0.5).mul_(0.5)


#inputs_lin = torch.tensor([[-1.,0,1,2,-2],[5, 4, 3, 2, 1]])/10
#weights_lin = torch.tensor([[-1.,0,1,2,-2],[5, 4, 3, 2, 1],[-1, -2, -3, -4, -5]])/10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(16,32,3, bias=False )
        self.conv1.weight.data = torch.clone(weights)

        self.linear = nn.Linear(288,10, bias = False)
#        print(self.linear.weight.data.shape)
        self.linear.weight.data = torch.clone(weights_lin)

    def forward(self, x):
        x = self.conv1(x)
#        x = x.view(x.size(0), -1)
#        x = self.linear(x)
        return x


class my_Net(nn.Module):
    def __init__(self):
        super(my_Net, self).__init__()
        self.conv1 = Conv2d_mvm(16,32,3, bit_slice = 2, bit_stream = 2, bias=False, input_bits=16, input_bit_frac=12, weight_bits=16, acm_bits=32, acm_bit_frac=20,  ind=ind)   # --> my custom module for mvm
        self.conv1.weight.data = torch.clone(weights)

        self.linear = Linear_mvm(288,10, bit_slice = 4, bit_stream = 2, bias=False, input_bits=32, weight_bits=32, acm_bits=32, ind=ind)
        self.linear.weight.data = torch.clone(weights_lin)

    def forward(self, x):
        x = self.conv1(x)
#        x = x.view(x.size(0),-1)
#        x = self.linear(x)
        return x


net = Net()
mynet = my_Net()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
mynet.to(device)
inputs = inputs.to(device)

result_net = net(inputs)
print(result_net[0:2,:2])

result_mynet = mynet(inputs)
#print(result_mynet[0:2,:2])

dif = result_net-result_mynet
print(dif[:2,:2])
print(torch.norm(result_net-result_mynet))

