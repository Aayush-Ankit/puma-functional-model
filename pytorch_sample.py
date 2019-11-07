import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_mvm_class import *

inputs = torch.tensor([[[[-1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]], [[[1.,0,1],[-2,1,0],[1,-2,1]],[[2,-3,1],[-2,0,-1],[4,-2,-1]],[[-3,2,1],[0,2,1],[-5,3,2]]]])/10
#inputs = torch.tensor([[[[1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]], [[[-1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]]])

labels = torch.tensor([1, 1])
weights = torch.tensor([[[[-2.,1],[-1,2]],[[-4,2],[0,1]],[[-1,0],[-3,-2]]],[[[2.,1],[1,2]],[[3,2],[1,1]],[[1,2],[3,2]]]])/10
trainloader = [[inputs, labels]]
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform =transforms.Compose([transforms.ToTensor()]))
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

inputs = torch.rand(2,16,5,5).mul_(2).sub_(1)
weights = torch.rand(32,16,2,2).sub_(0.5)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,2,2, bias=False)
        self.conv1.weight.data = torch.clone(weights)

    def forward(self, x):
        x = self.conv1(x)
        return x



class my_Net(nn.Module):
    def __init__(self):
        super(my_Net, self).__init__()
        self.conv1 = Conv2d_mvm(3,2,2, bias=False)   # --> my custom module for mvm
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


result_net = net(inputs)
result_mynet = mynet(inputs)
print(result_net)#[0,:2])
print(result_mynet)#[0,:2])
print(torch.norm(result_net-result_mynet))

