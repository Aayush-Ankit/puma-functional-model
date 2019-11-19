import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import pdb
import models
from pytorch_mvm_class_v2 import *
import os
import argparse
from data import get_dataset
from preprocess import get_transform
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
import torch.utils.data.distributed
from utils import *
from torchvision.utils import save_image
os.environ['CUDA_VISIBLE_DEVICES']= '2'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def accuracy(output, target, training, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if training:
        correct = pred.eq(target.data.view(1, -1).expand_as(pred))
    else:
        correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test():
    global best_acc
    flag = True
    training = False
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    for batch_idx,(data, target) in enumerate(testloader):
        target = target.cuda()
        data_var = torch.autograd.Variable(data.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

                                    
        output = model(data_var)
        loss= criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, training, topk=(1, 5))
        losses.update(loss.data, data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))


        if flag == True:
            if batch_idx % 1 == 0:
                print('[{0}/{1}({2:.0f}%)]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       batch_idx, len(testloader), 100. *float(batch_idx)/len(testloader),
                       loss=losses, top1=top1, top5=top5))
        else:
            if batch_idx % 1 == 0:
               print('Epoch: [{0}][{1}/{2}({3:.0f}%)]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, batch_idx, len(testloader), 100. *float(batch_idx)/len(testloader),
                       loss=losses, top1=top1, top5=top5))
        if batch_idx == 20:
            break


    acc = top1.avg
    # if acc > best_acc:
    #     best_acc = acc
    #     save_state(model, best_acc)
    

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return acc, losses.avg

def test_mvm():
    global best_acc
    flag = True
    training = False
    model_mvm.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx,(data, target) in enumerate(testloader):
        target = target.cuda()
        data_var = torch.autograd.Variable(data.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

                                    
        output = model_mvm(data_var)
        loss= criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, training, topk=(1, 5))
        losses.update(loss.data, data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))
        if flag == True:
            if batch_idx % 1 == 0:
                print('[{0}/{1}({2:.0f}%)]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       batch_idx, len(testloader), 100. *float(batch_idx)/len(testloader),
                       loss=losses, top1=top1, top5=top5))
        else:
            if batch_idx % 1 == 0:
               print('Epoch: [{0}][{1}/{2}({3:.0f}%)]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, batch_idx, len(testloader), 100. *float(batch_idx)/len(testloader),
                       loss=losses, top1=top1, top5=top5))
        if batch_idx == 20:
            break

    acc = top1.avg
    # if acc > best_acc:
    #     best_acc = acc
    #     save_state(model, best_acc)
    

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return acc, losses.avg



## To Indranil & Mustafa: This is for using 'for loops' in mvm_tensor. Just execute with '-i' at command line
# ind = False
# for i in range(len(sys.argv)):
#     if sys.argv[i] == '-i':
#         ind = True
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                         metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-i', default=False,
                         metavar='N', help='turn on Ind feature')
    parser.add_argument('--data', action='store', default='/local/scratch/a/imagenet/imagenet2012/',
            help='dataset path')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                help='dataset name or folder')
    parser.add_argument('--arch', action='store', default='resnet18_imnet',
        help='the architecture for the network: resnet')
    parser.add_argument('--model', '-a', metavar='MODEL', default='resnet18_imnet',
                choices=model_names,
                help='model architecture: ' +
                ' | '.join(model_names) +
                ' (default: resnet)')
    parser.add_argument('--pretrained', action='store', default=None,
        help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
        help='evaluate the model')
    parser.add_argument('--input_size', type=int, default=None,
                help='image input size')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 8)')
    parser.add_argument('-cuda', '--cuda_gpu', default=[0], type=int, metavar='N',
                help='gpu index (default: 8)')
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES']= str(args.cuda_gpu)
    
    if args.i == 'True':
        ind = True
    else:
        ind = False
    #



    print('==> building model',args.arch,'...')
    if args.arch == 'resnet18_imnet':
        #print(models.__dict__)
        model = models.__dict__[args.model]
        #model_config = {'input_size': args.input_size, 'dataset': args.dataset}
        print(model)
    else:
        raise Exception(args.arch+' is currently not supported')


    model = model()
    model_mvm = models.__dict__['resnet18_imnet_mvm']
    model_mvm = model_mvm(ind)
    #pdb.set_trace()

    print('==> Initializing model parameters ...')
    weights_conv = []
    weights_lin = []
    bn_data = []
    bn_bias = []
    running_mean = []
    running_var = []
    num_batches = []

    if not args.pretrained:
        for m in model.modules():
            
          #  print (m)for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                weights_conv.append(m.weight.data.clone())
            #print(m.weight.data)
            #raw_input()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                    stdv = 1. / math.sqrt(m.weight.data.size(1))
                    m.weight.data.uniform_(-stdv, stdv)
                    weights_lin.append(m.weight.data.clone())
                    if m.bias is not None:
                       m.bias.data.uniform_(-stdv, stdv)
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                weights_conv.append(m.weight.data.clone())
                #print(m.weight.data)
                #raw_input()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_data.append(m.weight.data.clone())
                bn_bias.append(m.bias.data.clone())
                running_mean.append(m.running_mean.data.clone())
                running_var.append(m.running_var.data.clone())
                num_batches.append(m.num_batches_tracked.clone())
            elif isinstance(m, nn.Linear):
                weights_lin.append(m.weight.data.clone())

    i=0
    j=0
    k=0
    for m in model_mvm.modules():
        
      #  print (m)for m in model.modules():
        if isinstance(m, Conv2d_mvm):
            m.weight.data = weights_conv[i]
            i = i+1
        #print(m.weight.data)
        #raw_input()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data = bn_data[j]
            m.bias.data = bn_bias[j]
            m.running_mean.data = running_mean[j]
            m.running_var.data = running_var[j]
            m.num_batches_tracked = num_batches[j]
            j = j+1
        elif isinstance(m, Linear_mvm):
            #pdb.set_trace()
            m.weight.data = weights_lin[k]
            k=k+1
    model.cuda()
    model_mvm.cuda()
    if len(args.cuda_gpu)>1:
        model = torch.nn.DataParallel(model.cuda(), device_ids=[0])
        model_mvm = torch.nn.DataParallel(model_mvm.cuda(), device_ids=[0])
    print('Near Data Loadin')
    traindir = os.path.join(args.data, 'train')
    print(traindir)
    valdir = os.path.join(args.data, 'val')
    print(valdir)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    torch_seed = 40#torch.initial_seed()
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,drop_last=True)


    print('Data Loading done')
    criterion = nn.CrossEntropyLoss()

    if args.evaluate:
        test()
        test_mvm()
        exit(0)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #net.to(device)
    # mynet.to(device)
    # inputs = inputs.to(device)

    # result_net = model(inputs)
    # print(result_net[0][0])#[0,:2])

    # result_mynet = model_mvm(inputs)
    # print(result_mynet[0][0])#[0,:2])
#print(torch.norm(result_net-result_mynet))

