import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_mvm_class import *

inputs = torch.tensor([[[[1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]]])
labels = torch.tensor([1])
weights = torch.tensor([[[[-2.,1],[1,2]],[[-4,2],[0,1]],[[1,0],[3,2]]],[[[2.,1],[1,2]],[[3,2],[1,1]],[[1,2],[3,2]]]])/10
trainloader = [[inputs, labels]]
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform =transforms.Compose([transforms.ToTensor()]))
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d_mvm(3,2,2, bias=False)#, padding = 1)   # --> my custom module for mvm
#        self.conv1 = nn.Conv2d(3,2,2, bias=False)
        self.conv1.weight.data = torch.clone(weights)
#        self.conv1.weight.requires_grad = True

        self.fc1 = nn.Linear(8, 2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        print(x)    # checking output of mvm operation. 
        x = x.view(-1, 8)
        x = self.fc1(x)
        return x


net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for itr in range(1):
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
       #inputs, labels = inputs.to(device), labels.to(device)    
        outputs = net(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
    
#        optimizer.step()
#        print(outputs)
