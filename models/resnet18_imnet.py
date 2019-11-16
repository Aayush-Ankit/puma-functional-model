import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
__all__ = ['resnet18_imnet']


  
class resnet(nn.Module):

    def __init__(self):
        super(resnet, self).__init__()

    def forward(self, x):
        #t = time.time()
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.bn1(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.bn2(x)
        # #x = self.maxpool1(x)

        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.bn3(x)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.bn4(x)
        # #x = self.maxpool2(x)

        # x = self.conv5(x)
        # #print('Conv 5', x[0][0][0])
        # x = F.relu(x)
        # x = self.bn5(x)
        # x = self.conv6(x)
        # x = F.relu(x)
        # x = self.bn6(x)
        # #x = self.maxpool3(x)

        # x = self.conv7(x)
        # #print('Conv7', x[0][0][0])
        # x = F.relu(x)
        # x = self.bn7(x)
        # x = self.conv8(x)
        # #print('Conv8', x[0][0][0])
        # x = F.relu(x)
        # x = self.bn8(x)
        # #x = self.maxpool4(x)


        # x = self.conv9(x)
        # #print('Conv9', x[0][0][0])
        # x = F.relu(x)
        # x = self.bn9(x)
        # x = self.conv10(x)
        # #print('Conv10', x[0][0][0])
        # x = F.relu(x)
        # x = self.bn10(x)
        # #x = self.maxpool5(x)

        # x = self.conv11(x)
        # #print('Conv11', x[0][0][0])
        # x = F.relu(x)
        # x = self.bn11(x)
        # x = self.conv12(x)
        # #print('Conv12', x[0][0][0])
        # x = F.relu(x)
        # x = self.bn12(x)
        # x = self.conv13(x)
        # #print('Conv13', x[0][0][0])
        # x = F.relu(x)
        # x = self.bn13(x)

        # #pdb.set_trace()
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # #pdb.set_trace()
        # x = self.linear(x)
        #print('Linear', x[0][0])
        # t1 = time.time()
        # print('Time taken: ',t1-t)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        residual1 = x.clone() 
        out = x.clone() 
        out = self.conv2(out)
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
        out = self.conv7(out)
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
        out = self.conv11(out)
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
        out = self.conv15(out)
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

        x = self.fc(x)

        x = self.bn19(x)

        x = self.logsoftmax(x)

        return x


class ResNet_imagenet(resnet):

    def __init__(self, num_classes=100):
        super(ResNet_imagenet, self).__init__()
        # self.conv1 = nn.Conv2d(3,16,3, stride = 1, padding=1, bias = False, bias=False, bias=False)	
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(64,64,3, stride = 1, padding=1, bias = False, bias=False, bias=False)
        # #self.conv2.weight.data = torch.clone(weights_conv2)
        # #self.maxpool1 = nn.MaxPool2d(2,2)

        # self.bn2 = nn.BatchNorm2d(64)
        # #self.bn2.running_mean.zero_() 
        # #self.bn2.running_var.fill_(1)


        # self.conv3 = nn.Conv2d(64,64,3, stride = 1, padding=1, bias = False, bias=False, bias=False)
        # #self.conv3.weight.data = torch.clone(weights_conv3)

        # self.bn3 = nn.BatchNorm2d(64)
        # #self.bn3.running_mean.zero_() 
        # #self.bn3.running_var.fill_(1)


        # self.conv4 = nn.Conv2d(64,64,3, stride = 1, padding=1, bias = False, bias=False, bias=False)
        # #self.conv4.weight.data = torch.clone(weights_conv4)

        # self.bn4 = nn.BatchNorm2d(64)
        # #self.bn4.running_mean.zero_() 
        # #self.bn4.running_var.fill_(1)

        # self.maxpool2 = nn.MaxPool2d(2,2)

        # self.conv5 = nn.Conv2d(64,64,3, stride = 1, padding=1, bias = False, bias=False, bias=False)
        # #self.conv5.weight.data = torch.clone(weights_conv5)

        # self.bn5 = nn.BatchNorm2d(64)
        # #self.bn5.running_mean.zero_() 
        # #self.bn5.running_var.fill_(1)


        # self.conv6= nn.Conv2d(64,128,3, stride = 2, padding=1, bias = False, bias=False, bias=False)
        # #self.conv6.weight.data = torch.clone(weights_conv6)

        # self.bn6= nn.BatchNorm2d(128)
        # #self.bn6.running_mean.zero_() 
        # #self.bn6.running_var.fill_(1)


        # self.conv7 = nn.Conv2d(128,128,3, stride = 1, padding=1, bias = False, bias=False, bias=False)
        # #self.conv7.weight.data = torch.clone(weights_conv7)
        
        # self.bn7 = nn.BatchNorm2d(128)
        # #self.bn7.running_mean.zero_() 
        # #self.bn7.running_var.fill_(1)
        
        # #self.maxpool3 = nn.MaxPool2d(2,2)
        
        
        # self.conv8 = nn.Conv2d(128,128,3, stride = 1, padding=1, bias = False, bias=False, bias=False)
        # #self.conv8.weight.data = torch.clone(weights_conv8)
        
        # self.bn8 = nn.BatchNorm2d(128)
        # #self.bn8.running_mean.zero_() 
        # #self.bn8.running_var.fill_(1)
        
        # self.conv9 = nn.Conv2d(128,128,3, stride = 1, padding=1, bias = False, bias=False, bias=False)
        # #self.conv9.weight.data = torch.clone(weights_conv9)
        
        # self.bn9 = nn.BatchNorm2d(128)
        # #self.bn9.running_mean.zero_() 
        # #self.bn9.running_var.fill_(1)
        
        # #self.maxpool4 = nn.MaxPool2d(2,2)
        
        # self.conv10 = nn.Conv2d(128,256,3, stride = 2, padding=1, bias = False, bias=False, bias=False)
        # #self.conv10.weight.data = torch.clone(weights_conv10)
        
        # self.bn10 = nn.BatchNorm2d(256)
        # #self.bn10.running_mean.zero_() 
        # #self.bn10.running_var.fill_(1)

        # self.conv11 = nn.Conv2d(256,256,3, stride = 1, padding=1, bias = False, bias=False, bias=False)
        # #self.conv11.weight.data = torch.clone(weights_conv11)
        
        # self.bn11 = nn.BatchNorm2d(256)
        # #self.bn11.running_mean.zero_() 
        # #self.bn11.running_var.fill_(1)

        # self.conv12 = nn.Conv2d(256,256,3, stride = 1, padding=1, bias = False, bias=False, bias=False)
        # #self.conv12.weight.data = torch.clone(weights_conv12)
        
        # self.bn12 = nn.BatchNorm2d(256)
        # #self.bn12.running_mean.zero_() 
        # #self.bn12.running_var.fill_(1)

        # self.conv13 = nn.Conv2d(256,256,3, stride = 1, padding=1, bias = False, bias=False, bias=False)
        # #self.conv13.weight.data = torch.clone(weights_conv13)
        
        # self.bn13 = nn.BatchNorm2d(256)
        # #self.bn13.running_mean.zero_() 
        # #self.bn13.running_var.fill_(1)
        
        # self.avgpool = nn.AvgPool2d(8)
        
        # self.linear = nn.Linear(256,10, bias = False)
        self.inflate = 1
        self.conv1=nn.Conv2d(3,int(64*self.inflate), kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1= nn.BatchNorm2d(int(64*self.inflate))
        self.relu1=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2=nn.Conv2d(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn2= nn.BatchNorm2d(int(64*self.inflate))
        self.relu2=nn.ReLU(inplace=True)
        #######################################################

        self.conv3=nn.Conv2d(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn3= nn.BatchNorm2d(int(64*self.inflate))
        self.relu3=nn.ReLU(inplace=True)
        #######################################################

        self.conv4=nn.Conv2d(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn4= nn.BatchNorm2d(int(64*self.inflate))
        self.relu4=nn.ReLU(inplace=True)
        #######################################################

        self.conv5=nn.Conv2d(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn5= nn.BatchNorm2d(int(64*self.inflate))
        self.relu5=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.conv6=nn.Conv2d(int(64*self.inflate), int(128*self.inflate), kernel_size=3, stride=2, padding=1, bias = False)
        self.bn6= nn.BatchNorm2d(int(128*self.inflate))
        self.resconv1=nn.Sequential(nn.Conv2d(int(64*self.inflate), int(128*self.inflate), kernel_size=1, stride=2, padding=0, bias = False),
        nn.BatchNorm2d(int(128*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu6=nn.ReLU(inplace=True)
        #######################################################

        self.conv7=nn.Conv2d(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn7= nn.BatchNorm2d(int(128*self.inflate))
        self.relu7=nn.ReLU(inplace=True)
        #######################################################

        self.conv8=nn.Conv2d(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn8= nn.BatchNorm2d(int(128*self.inflate))
        self.relu8=nn.ReLU(inplace=True)
        #######################################################

        self.conv9=nn.Conv2d(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn9= nn.BatchNorm2d(int(128*self.inflate))
        self.relu9=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.conv10=nn.Conv2d(int(128*self.inflate), int(256*self.inflate), kernel_size=3, stride=2, padding=1, bias = False)
        self.bn10= nn.BatchNorm2d(int(256*self.inflate))
        self.resconv2=nn.Sequential(nn.Conv2d(int(128*self.inflate), int(256*self.inflate), kernel_size=1, stride=2, padding=0, bias = False),
        nn.BatchNorm2d(int(256*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu10=nn.ReLU(inplace=True)
        #######################################################

        self.conv11=nn.Conv2d(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn11= nn.BatchNorm2d(int(256*self.inflate))
        self.relu11=nn.ReLU(inplace=True)
        #######################################################

        self.conv12=nn.Conv2d(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn12= nn.BatchNorm2d(int(256*self.inflate))
        self.relu12=nn.ReLU(inplace=True)
        #######################################################

        self.conv13=nn.Conv2d(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn13= nn.BatchNorm2d(int(256*self.inflate))
        self.relu13=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.conv14=nn.Conv2d(int(256*self.inflate), int(512*self.inflate), kernel_size=3, stride=2, padding=1, bias = False)
        self.bn14= nn.BatchNorm2d(int(512*self.inflate))
        self.resconv3=nn.Sequential(nn.Conv2d(int(256*self.inflate), int(512*self.inflate), kernel_size=1, stride=2, padding=0, bias = False),
        nn.BatchNorm2d(int(512*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu14=nn.ReLU(inplace=True)
        #######################################################

        self.conv15=nn.Conv2d(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn15= nn.BatchNorm2d(int(512*self.inflate))
        self.relu15=nn.ReLU(inplace=True)
        #######################################################

        self.conv16=nn.Conv2d(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn16= nn.BatchNorm2d(int(512*self.inflate))
        self.relu16=nn.ReLU(inplace=True)
        #######################################################

        self.conv17=nn.Conv2d(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn17= nn.BatchNorm2d(int(512*self.inflate))
        self.relu17=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.avgpool=nn.AvgPool2d(7)
        self.bn18= nn.BatchNorm1d(int(512*self.inflate))
        self.fc=nn.Linear(int(512*self.inflate),num_classes, bias = False)
        self.bn19= nn.BatchNorm1d(1000)
        self.logsoftmax=nn.LogSoftmax()

        #        #print(self.linear.weight.data.shape)
        #self.linear.weight.data = torch.clone(weights_lin)


	

        #init_model(self)
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-1,
        #        'weight_decay': 1e-4, 'momentum': 0.9},
        #    81: {'lr': 1e-4},
        #    122: {'lr': 1e-5, 'weight_decay': 0},
        #    164: {'lr': 1e-6}
        #}

def resnet18_imnet(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    num_classes = 1000
    return ResNet_imagenet(num_classes=num_classes)
    #if dataset == 'cifar100':
        
        
