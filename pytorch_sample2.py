import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from pytorch_mvm_class import *

inputs = torch.tensor([[[[-1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]], [[[1.,0,1],[-2,1,0],[1,-2,1]],[[2,-3,1],[-2,0,-1],[4,-2,-1]],[[-3,2,1],[0,2,1],[-5,3,2]]]])/10
#inputs = torch.tensor([[[[1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]], [[[-1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]]])

labels = torch.tensor([1, 1])
weights = torch.tensor([[[[-2.,1],[-1,2]],[[-4,2],[0,1]],[[-1,0],[-3,-2]]],[[[2.,1],[1,2]],[[3,2],[1,1]],[[1,2],[3,2]]]])/10
trainloader = [[inputs, labels]]
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform =transforms.Compose([transforms.ToTensor()]))
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

inputs = torch.rand(64,128,32,32).sub(0.5).mul(0.5)
weights = torch.rand(256,128,3,3).sub(0.5).mul(0.5)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(256,512,3, bias=False)
        self.conv1.weight.data = torch.clone(weights)

    def forward(self, x):
        x = self.conv1(x)
        return x



class my_Net(nn.Module):
    def __init__(self):
        super(my_Net, self).__init__()
        self.conv1 = Conv2d_mvm(256,512,3, bias=False)   # --> my custom module for m
        self.conv1.weight.data = torch.clone(weights)

    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()
mynet = my_Net()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
mynet.to(device)
inputs = inputs.to(device)

torch.cuda.synchronize()
begin = time.time()
result_net = net(inputs)

torch.cuda.synchronize()
end = time.time()
#print(end-begin)
result_mynet = mynet(inputs)

torch.cuda.synchronize()
end2 = time.time()
print(end2-end)

print(result_net[0][0][0])#[0,:2])
print(result_mynet[0][0][0])#[0,:2])
print(torch.norm(result_net-result_mynet))

