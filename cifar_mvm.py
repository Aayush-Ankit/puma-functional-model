import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from pytorch_mvm_class import *

#inputs = torch.tensor([[[[1.,0,1],[-2,1,0],[1,2,1]],[[2,3,1],[-2,0,1],[-4,2,1]],[[3,2,1],[0,2,1],[-5,3,2]]]])
#labels = torch.tensor([1])
#weights = torch.tensor([[[[-2.,1],[1,-2]],[[4,2],[0,1]],[[1,0],[3,2]]],[[[2.,1],[1,2]],[[3,2],[1,1]],[[1,-2],[-3,2]]]])/10
#trainloader = [[inputs, labels]]
normalize = transforms.Normalize( mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) 
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform =transforms.Compose([transforms.ToTensor(), normalize]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d_mvm(3,16,3, padding = 1)
#        self.conv1 = nn.Conv2d(3,16,3, padding=1)
#        self.conv1.weight.data = torch.clone(weights)
#        self.conv1.weight.requires_grad = True
        
        self.fc1 = nn.Linear(16384, 100, bias=False)
        for m in self.modules():
            if isinstance(m, Conv2d_mvm) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                std = np.sqrt(2./n)
                m.weight.data.normal_(0, std)
                m.bias.data.zero_()
#                print("normalized")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.conv1(x)
#        print(x.shape)
#        x = x.to(device)
        print(x)
        x = x.view(-1, 16384)
        x = self.fc1(x)
        return x


net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
print(net.conv1.weight.shape)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for itr in range(10):
    print(itr)
    for i, data in enumerate(trainloader):
        begin = time.time()
        # get the inputs; data is a list of [inputs, labels]
#        if i%1 == 0:
#            print(i)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)    
        outputs = net(inputs)
        forward = time.time()
#        print("forward: ",forward-begin)
        loss = criterion(outputs, labels)
        loss.backward()
        backward = time.time()
#        print("backward: ", backward-forward)
         

        optimizer.step()
        update = time.time()
#        print("update: ", update-backward)

#        print(outputs)
        print(loss)
