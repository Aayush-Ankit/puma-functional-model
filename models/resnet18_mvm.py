import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
from src.pytorch_mvm_class_v3 import *
__all__ = ['net']


  
class resnet(nn.Module):

    def __init__(self):
        super(resnet, self).__init__()

    def forward(self, x):
        t = time.time()
        x = self.conv1(x)
        print('Conv1: ', torch.mean(x))
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        residual1 = x.clone() 
        out = x.clone() 
        t1 = time.time()
        out = self.conv2(out)
        t2 = time.time()
        print('Time taken - Conv2 - 64: ',t2-t1)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out+=residual1
        out = F.relu(out)
        residual1 = out.clone() 
        ################################### 
        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out+=residual1
        out = F.relu(out)
        residual1 = out.clone() 
        ################################### 
        #########Layer################ 
        out = self.conv6(out)
        out = self.bn6(out)
        residual1 = self.resconv1(residual1)
        out = F.relu(out)
        t3 = time.time()
        out = self.conv7(out)
        t4 = time.time()
        print('Time taken - Conv7 - 128: ',t4-t3)
        out = self.bn7(out)
        out+=residual1
        out = F.relu(out)
        residual1 = out.clone() 
        ################################### 
        out = self.conv8(out)
        out = self.bn8(out)
        out = F.relu(out)
        out = self.conv9(out)
        out = self.bn9(out)
        out+=residual1
        out = F.relu(out)
        residual1 = out.clone() 
        ################################### 
        #########Layer################ 
        out = self.conv10(out)
        out = self.bn10(out)
        residual1 = self.resconv2(residual1)
        out = F.relu(out)
        t3 = time.time()
        out = self.conv11(out)
        t4 = time.time()
        print('Time taken - Conv11 -256: ',t4-t3)
        out = self.bn11(out)
        out+=residual1
        out = F.relu(out)
        residual1 = out.clone() 
        ################################### 
        out = self.conv12(out)
        out = self.bn12(out)
        out = F.relu(out)
        out = self.conv13(out)
        out = self.bn13(out)
        out+=residual1
        out = F.relu(out)
        residual1 = out.clone() 
        ################################### 
        #########Layer################ 
        out = self.conv14(out)
        out = self.bn14(out)
        residual1 = self.resconv3(residual1)
        out = F.relu(out)
        t3 = time.time()
        out = self.conv15(out)
        t4 = time.time()
        print('Time taken - Conv15 - 512: ',t4-t3)
        out = self.bn15(out)
        out+=residual1
        out = F.relu(out)
        residual1 = out.clone() 
        ################################### 
        out = self.conv16(out)
        out = self.bn16(out)
        out = F.relu(out)
        out = self.conv17(out)
        out = self.bn17(out)
        out+=residual1
        out = F.relu(out)
        residual1 = out.clone() 
        ################################### 
        #########Layer################ 
        x=out 
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.bn18(x)
        t3 = time.time()
        x = self.fc(x)
        t4 = time.time()
        print('Time taken - Linear - 512: ',t4-t3)

        x = self.bn19(x)

        x = self.logsoftmax(x)
        t5 = time.time()
        print('Total Time taken: ',t5-t)

        return x


class ResNet_imagenet(resnet):

    def __init__(self, ind, num_classes=100):
        super(ResNet_imagenet, self).__init__()
        self.inflate = 1
        wbit_frac = 6
        ibit_frac = 6
        wbit_total = 8
        ibit_total = 8
        bit_slice_in = 4
        bit_stream_in = 4
        self.conv1=Conv2d_mvm(3,int(64*self.inflate), kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1= nn.BatchNorm2d(int(64*self.inflate))
        self.relu1=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2=Conv2d_mvm(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2= nn.BatchNorm2d(int(64*self.inflate))
        self.relu2=nn.ReLU(inplace=True)
        #######################################################

        self.conv3=Conv2d_mvm(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3= nn.BatchNorm2d(int(64*self.inflate))
        self.relu3=nn.ReLU(inplace=True)
        #######################################################

        self.conv4=Conv2d_mvm(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4= nn.BatchNorm2d(int(64*self.inflate))
        self.relu4=nn.ReLU(inplace=True)
        #######################################################

        self.conv5=Conv2d_mvm(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5= nn.BatchNorm2d(int(64*self.inflate))
        self.relu5=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.conv6=Conv2d_mvm(int(64*self.inflate), int(128*self.inflate), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6= nn.BatchNorm2d(int(128*self.inflate))
        self.resconv1=nn.Sequential(Conv2d_mvm(int(64*self.inflate), int(128*self.inflate), kernel_size=1, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(int(128*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu6=nn.ReLU(inplace=True)
        #######################################################

        self.conv7=Conv2d_mvm(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7= nn.BatchNorm2d(int(128*self.inflate))
        self.relu7=nn.ReLU(inplace=True)
        #######################################################

        self.conv8=Conv2d_mvm(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8= nn.BatchNorm2d(int(128*self.inflate))
        self.relu8=nn.ReLU(inplace=True)
        #######################################################

        self.conv9=Conv2d_mvm(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9= nn.BatchNorm2d(int(128*self.inflate))
        self.relu9=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.conv10=Conv2d_mvm(int(128*self.inflate), int(256*self.inflate), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn10= nn.BatchNorm2d(int(256*self.inflate))
        self.resconv2=nn.Sequential(Conv2d_mvm(int(128*self.inflate), int(256*self.inflate), kernel_size=1, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(int(256*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu10=nn.ReLU(inplace=True)
        #######################################################

        self.conv11=Conv2d_mvm(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11= nn.BatchNorm2d(int(256*self.inflate))
        self.relu11=nn.ReLU(inplace=True)
        #######################################################

        self.conv12=Conv2d_mvm(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12= nn.BatchNorm2d(int(256*self.inflate))
        self.relu12=nn.ReLU(inplace=True)
        #######################################################

        self.conv13=Conv2d_mvm(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13= nn.BatchNorm2d(int(256*self.inflate))
        self.relu13=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.conv14=Conv2d_mvm(int(256*self.inflate), int(512*self.inflate), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn14= nn.BatchNorm2d(int(512*self.inflate))
        self.resconv3=nn.Sequential(Conv2d_mvm(int(256*self.inflate), int(512*self.inflate), kernel_size=1, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(int(512*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu14=nn.ReLU(inplace=True)
        #######################################################

        self.conv15=Conv2d_mvm(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15= nn.BatchNorm2d(int(512*self.inflate))
        self.relu15=nn.ReLU(inplace=True)
        #######################################################

        self.conv16=Conv2d_mvm(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16= nn.BatchNorm2d(int(512*self.inflate))
        self.relu16=nn.ReLU(inplace=True)
        #######################################################

        self.conv17=Conv2d_mvm(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17= nn.BatchNorm2d(int(512*self.inflate))
        self.relu17=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.avgpool=nn.AvgPool2d(7)
        self.bn18= nn.BatchNorm1d(int(512*self.inflate))
        self.fc=Linear_mvm(int(512*self.inflate),num_classes, bias=False)
        self.bn19= nn.BatchNorm1d(1000)
        self.logsoftmax=nn.LogSoftmax()


def net(ind, **kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    num_classes = 1000
    return ResNet_imagenet(ind, num_classes=num_classes)
    #if dataset == 'cifar100':
        
        
