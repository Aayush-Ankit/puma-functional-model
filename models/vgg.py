import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
__all__ = ['vgg']


  
class vgg16(nn.Module):

    def __init__(self):
        super(vgg16, self).__init__()

    def forward(self, x):
	    t = time.time()
	    x = self.conv1(x)
	    x = F.relu(x)
	    x = self.bn1(x)
	    x = self.conv2(x)
	    x = F.relu(x)
	    x = self.bn2(x)
	    #x = self.maxpool1(x)
	    
	    x = self.conv3(x)
	    x = F.relu(x)
	    x = self.bn3(x)
	    x = self.conv4(x)
	    x = F.relu(x)
	    x = self.bn4(x)
	    #x = self.maxpool2(x)

	    x = self.conv5(x)
	    x = F.relu(x)
	    x = self.bn5(x)
	    x = self.conv6(x)
	    x = F.relu(x)
	    x = self.bn6(x)
	    #x = self.maxpool3(x)
	    
	    x = self.conv7(x)
	    x = F.relu(x)
	    x = self.bn7(x)
	    x = self.conv8(x)
	    x = F.relu(x)
	    x = self.bn8(x)
	    #x = self.maxpool4(x)
	    
	    
	    x = self.conv9(x)
	    x = F.relu(x)
	    x = self.bn9(x)
	    x = self.conv10(x)
	    x = F.relu(x)
	    x = self.bn10(x)
	    #x = self.maxpool5(x)
	    
	    x = self.conv11(x)
	    x = F.relu(x)
	    x = self.bn11(x)
	    x = self.conv12(x)
	    x = F.relu(x)
	    x = self.bn12(x)
	    x = self.conv13(x)
	    x = F.relu(x)
	    x = self.bn13(x)

	    #pdb.set_trace()
	    x = self.avgpool(x)
	    x = x.view(x.size(0), -1)
	    #pdb.set_trace()
	    x = self.linear(x)
	    t1 = time.time()
	    print('Time taken: ',t1-t)
	    return x

class VGG_cifar100(vgg16):

    def __init__(self, num_classes=100):
        super(VGG_cifar100, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3, bias=False)
        #self.conv1.weight.data = torch.clone(weights_conv1)
	
        self.bn1 = nn.BatchNorm2d(64)
        #self.bn1.running_mean.zero_()
        #self.bn1.running_var.fill_(1)


        self.conv2 = nn.Conv2d(64,64,3, bias=False)
        #self.conv2.weight.data = torch.clone(weights_conv2)
        #self.maxpool1 = nn.MaxPool2d(2,2)

        self.bn2 = nn.BatchNorm2d(64)
        #self.bn2.running_mean.zero_() 
        #self.bn2.running_var.fill_(1)


        self.conv3 = nn.Conv2d(64,64,3, bias=False)
        #self.conv3.weight.data = torch.clone(weights_conv3)

        self.bn3 = nn.BatchNorm2d(64)
        #self.bn3.running_mean.zero_() 
        #self.bn3.running_var.fill_(1)


        self.conv4 = nn.Conv2d(64,64,3, bias=False)
        #self.conv4.weight.data = torch.clone(weights_conv4)

        self.bn4 = nn.BatchNorm2d(64)
        #self.bn4.running_mean.zero_() 
        #self.bn4.running_var.fill_(1)

        self.maxpool2 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(64,64,3, bias=False)
        #self.conv5.weight.data = torch.clone(weights_conv5)

        self.bn5 = nn.BatchNorm2d(64)
        #self.bn5.running_mean.zero_() 
        #self.bn5.running_var.fill_(1)


        self.conv6= nn.Conv2d(64,128,3, bias=False)
        #self.conv6.weight.data = torch.clone(weights_conv6)

        self.bn6= nn.BatchNorm2d(128)
        #self.bn6.running_mean.zero_() 
        #self.bn6.running_var.fill_(1)


        self.conv7 = nn.Conv2d(128,128,3, bias=False)
        #self.conv7.weight.data = torch.clone(weights_conv7)
        
        self.bn7 = nn.BatchNorm2d(128)
        #self.bn7.running_mean.zero_() 
        #self.bn7.running_var.fill_(1)
        
        #self.maxpool3 = nn.MaxPool2d(2,2)
        
        
        self.conv8 = nn.Conv2d(128,128,3, bias=False)
        #self.conv8.weight.data = torch.clone(weights_conv8)
        
        self.bn8 = nn.BatchNorm2d(128)
        #self.bn8.running_mean.zero_() 
        #self.bn8.running_var.fill_(1)
        
        self.conv9 = nn.Conv2d(128,128,3, bias=False)
        #self.conv9.weight.data = torch.clone(weights_conv9)
        
        self.bn9 = nn.BatchNorm2d(128)
        #self.bn9.running_mean.zero_() 
        #self.bn9.running_var.fill_(1)
        
        #self.maxpool4 = nn.MaxPool2d(2,2)
        
        self.conv10 = nn.Conv2d(128,256,3, bias=False)
        #self.conv10.weight.data = torch.clone(weights_conv10)
        
        self.bn10 = nn.BatchNorm2d(256)
        #self.bn10.running_mean.zero_() 
        #self.bn10.running_var.fill_(1)

        self.conv11 = nn.Conv2d(256,256,3, bias=False)
        #self.conv11.weight.data = torch.clone(weights_conv11)
        
        self.bn11 = nn.BatchNorm2d(256)
        #self.bn11.running_mean.zero_() 
        #self.bn11.running_var.fill_(1)

        self.conv12 = nn.Conv2d(256,256,3, bias=False)
        #self.conv12.weight.data = torch.clone(weights_conv12)
        
        self.bn12 = nn.BatchNorm2d(256)
        #self.bn12.running_mean.zero_() 
        #self.bn12.running_var.fill_(1)

        self.conv13 = nn.Conv2d(256,256,3, bias=False)
        #self.conv13.weight.data = torch.clone(weights_conv13)
        
        self.bn13 = nn.BatchNorm2d(256)
        #self.bn13.running_mean.zero_() 
        #self.bn13.running_var.fill_(1)
        
        self.avgpool = nn.AvgPool2d(6)
        
        self.linear = nn.Linear(256,10, bias = False)
        #        print(self.linear.weight.data.shape)
        #self.linear.weight.data = torch.clone(weights_lin)


	

        #init_model(self)
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-1,
        #        'weight_decay': 1e-4, 'momentum': 0.9},
        #    81: {'lr': 1e-4},
        #    122: {'lr': 1e-5, 'weight_decay': 0},
        #    164: {'lr': 1e-6}
        #}

def vgg(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    num_classes = 100
    return VGG_cifar100(num_classes=num_classes)
    #if dataset == 'cifar100':
        
        
