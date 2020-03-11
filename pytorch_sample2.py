import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from pytorch_mvm_class_v2 import *

#inputs = torch.tensor([[[[-1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]], [[[1.,0,1],[-2,1,0],[1,-2,1]],[[2,-3,1],[-2,0,-1],[4,-2,-1]],[[-3,2,1],[0,2,1],[-5,3,2]]]])/10
#inputs = torch.tensor([[[[1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]], [[[-1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]]])

labels = torch.tensor([0, 1, 0])
#weights = torch.tensor([[[[-2.,1],[-1,2]],[[-4,2],[0,1]],[[-1,0],[-3,-2]]],[[[2.,1],[1,2]],[[3,2],[1,1]],[[1,2],[3,2]]]])/10
#trainloader = [[inputs, labels]]
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform =transforms.Compose([transforms.ToTensor()]))
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

inputs = torch.rand(3,16,5,5).sub(0.5).mul(0.5)
weights = torch.rand(16,16,3,3).sub(0.5).mul(0.5)
l_weights = torch.rand(2,144).sub(0.5).mul(0.5)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3, bias=False)
        self.conv1.weight.data = torch.clone(weights)
       
        self.linear = nn.Linear(144, 2, bias=False) 
        self.linear.weight.data = torch.clone(l_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0),-1)
        x=self.linear(x)

        return x



class my_Net(nn.Module):
    def __init__(self):
        super(my_Net, self).__init__()
        self.conv1 = Conv2d_mvm(3,16,3, input_bits = 16, weight_bits = 16, bias=False)   # --> my custom module for m
        self.conv1.weight.data = torch.clone(weights)

        self.linear = Linear_mvm(144,2, bias = False)
        
#        self.linear = nn.Linear(144, 2, bias=False) 
        self.linear.weight.data = torch.clone(l_weights)

    def forward(self, x):
        x = self.conv1(x)
        x=x.view(x.size(0),-1)
        x = self.linear(x)

        return x


net = Net()
mynet = my_Net()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
mynet.to(device)
inputs = inputs.to(device)
labels = labels.to(device)

#torch.cuda.synchronize()
#begin = time.time()
result_net = net(inputs)

#torch.cuda.synchronize()
#end = time.time()
#print(end-begin)
result_mynet = mynet(inputs)

#torch.cuda.synchronize()
#end2 = time.time()
#print(end2-end)

#print(result_net[0][0][0])#[0,:2])
#print(result_mynet[0][0][0])#[0,:2])
#print(torch.norm(result_net-result_mynet))
#print(result_net.shape)

print(result_net)
print(result_mynet)

criterion = nn.CrossEntropyLoss()
optimizer_net = optim.SGD(net.parameters(), lr=0.1, momentum=0, weight_decay=0)
optimizer_mynet = optim.SGD(mynet.parameters(), lr=0.1, momentum=0, weight_decay=0)

optimizer_net.zero_grad()

for i in range(10):
    result_net = net(inputs)
    result_mynet = mynet(inputs)

    loss = criterion(result_net, labels)
    loss.backward()
    optimizer_net.step()

    loss_my = criterion(result_mynet, labels)
    loss_my.backward()
    optimizer_mynet.step()
    print(loss, loss_my)

result_net = net(inputs)
result_mynet = mynet(inputs)
print(result_net)
print(result_mynet)

