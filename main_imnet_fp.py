from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import utils
import torch.nn as nn
import torch.optim as optim
import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
import torch.utils.data.distributed
from preprocess import get_transform
import time
import logging
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import models
from data import get_dataset
from preprocess import get_transform
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from utils import *
import math
import numpy
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUDA_VISIBLE_DEVICES']='2'
#os.environ['CUDA_VISIBLE_DEVICES']='0'
#from models import resnet
from torch.autograd import Variable
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
#print('printing models dict %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#for name in models.__dict__ :
#  print(name)


#print('printing model names %%%%%%%%%%%%%%%%%%%%%%%%%%')
#print(model_names)
def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/resnet18_imnet_fp.pth.tar')
def accuracy(output, target, training, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print(target)
    #print(output)
    #print(target.view(1, -1).expand_as(pred))
    #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    if training:
    	correct = pred.eq(target.data.view(1, -1).expand_as(pred))
    else:
        correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(epoch,model):
    global best_train_acc
    best_train_acc=0
    model.train()
    training = True
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    correct = 0
    for batch_idx, (inputs, target) in enumerate(trainloader):
        # process the weights including binarization
        #bin_op.binarization()
    #t0 = time.time()
        if args.gpus is not None:
            inputs, target = inputs.cuda(), target.cuda()
            #target = target.cuda(async=True)
        #inputs = inputs.cuda(async=True)
        data = Variable(inputs)
        target = Variable(target)
    #print('Time taken for data_loading',time.time()-t0)
        
        # forwarding
       # data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        #print(model)
    #print(data)
        output = model(data)
        
        # backwarding
    #print(criterion)
    #print(output)
    #print(target)
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        #bin_op.restore()
        #bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        prec1, prec5 = accuracy(output.data, target, training, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
       # pred = output.data.max(1, keepdim=True)[1]
       # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    	
    	#save_state(model, best_train,acc)
        if batch_idx % 10 == 0:
            print('Epoch: [{0}][{1}/{2}({3:.0f}%)]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\tLR: {LR}'.format(
                   epoch, batch_idx, len(trainloader), 100. *float(batch_idx)/len(trainloader),
                   loss=losses, top1=top1, top5=top5, LR = optimizer.param_groups[0]['lr']))

    acc = top1.avg
    if acc > best_train_acc:
        	best_train_acc = acc
    print('Train Epoch: {}\t({:.2f}%)]\tLoss: {:.6f}\n'.format(
                epoch,
                acc, losses.avg))

    print('Best Accuracy: {:.2f}%\n'.format(best_train_acc))
    return acc, losses.avg

def test():
    global best_acc
    flag = False
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #bin_op.binarization()
    for batch_idx,(data, target) in enumerate(testloader):
        target = target.cuda()
        data_var = torch.autograd.Variable(data.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        #data_var, target_var = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model(data_var)
        loss= criterion(output, target_var)
        #test_loss += criterion(output, target).data[0]
       # pred = output.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target, flag, topk=(1, 5))
        losses.update(loss.data, data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        if batch_idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}({3:.0f}%)]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, batch_idx, len(testloader), 100. *float(batch_idx)/len(testloader),
                       loss=losses, top1=top1, top5=top5))
    #bin_op.restore()
    acc = top1.avg
    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)

    #test_loss /= len(testloader.dataset)
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    #    test_loss * 128., correct, len(testloader.dataset),
    #    100. * correct / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return acc, losses.avg
def adjust_learning_rate(optimizer, epoch):
    update_list = [30, 60, 90,95]
    lr_list = [1e-2, 1e-3, 1e-4, 1e-5]
    wd_list = [1e-4, 1e-4, 1e-4 , 1e-4]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            index_epoch = update_list.index(epoch)
            param_group['lr'] = lr_list[index_epoch]
            param_group['weight_decay'] = wd_list[index_epoch]
    return

if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='/local/scratch/a/imagenet/imagenet2012/',
            help='dataset path')
    parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
    parser.add_argument('--arch', action='store', default='resnet18_imnet',
            help='the architecture for the network: resnet')
    parser.add_argument('--model', '-a', metavar='MODEL', default='resnet18_imnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet)')

    parser.add_argument('--lr', action='store', default='0.1',
            help='the intial learning rate')
    parser.add_argument('--momentum', action='store', default='0.9',
            help='momentum')
    parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')

    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
    	                        
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')

    parser.add_argument('--gpus', default=[1],
                    help='gpus used for training - e.g 0,1,3')
    parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
    parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')

    args = parser.parse_args()
    print('==> Options:',args)
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'txt', results_file % 'html')

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'resnet_fp' or 'resnet_fpunrolled' or 'resnet18_imnet':
        #print(models.__dict__)
        model = models.__dict__[args.model]
        model_config = {'input_size': args.input_size, 'dataset': args.dataset}
        print(model,model_config)
    else:
        raise Exception(args.arch+' is currently not supported')

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    print(model)

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        best_train_acc=0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.data.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                weights_lin = m.weight.data.clone()
                if m.bias is not None:
                   m.bias.data.uniform_(-stdv, stdv)
            
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])
    model.cuda()
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model.cuda(), device_ids=[0,1])
        #model = torch.nn.DataParallel(model, args.gpus)

        #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)
     # prepare the data
    # print(args.data+'train')
    # if not os.path.isfile(args.data+'train'):
    #     # check the data path
    #     raise Exception\
    #             ('Please assign the correct data path with --data <DATA_PATH>')
    # default_transform = {
    #     'train': get_transform(args.dataset,
    #                            input_size=args.input_size, augment=True),
    #     'eval': get_transform(args.dataset,
    #                           input_size=args.input_size, augment=False)
    # }
    # transform = getattr(model, 'input_transform', default_transform)
    # train_data = get_dataset(args.dataset, 'train', transform['train'])

    # torch_seed = 40#torch.initial_seed()
    # def _init_fn(worker_id):
    #     numpy.random.seed(torch_seed)
    #     random.seed(torch_seed)
    #     numpy.random.seed(torch_seed)
    #     torch.manual_seed(torch_seed)
    #     torch.cuda.manual_seed_all(torch_seed)
    #     os.environ['PYTHONHASHSEED'] = str(torch_seed)

    # trainloader = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True,worker_init_fn=_init_fn)


    # #trainset = data.dataset(root=args.data, train=True)
    # #trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
    #        # shuffle=True, num_workers=2)
    # test_data = get_dataset(args.dataset, 'val', transform['eval'])
    # testloader = torch.utils.data.DataLoader(
    #     test_data,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True,worker_init_fn = _init_fn)
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
    # testset = data.dataset(root=args.data, train=False)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100,
          #  shuffle=False, num_workers=2)

    # define classes
    #classes = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100)

    # define solver and criterion
    base_lr = float(args.lr)

    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr, 
            'weight_decay':0.0001}]
    #optimizer = optim.Adam(params,lr=0.10,weight_decay=0.00001)
        optimizer = optim.SGD(params, momentum = float(args.momentum), lr=0.10,weight_decay=0.0001, nesterov=True, dampening=0)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    #bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)

    # start training
    for epoch in range(1, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        [trainacc,train_loss] = train(epoch,model)
        [testacc,test_loss] = test()
        results.add(epoch=epoch, train_loss=train_loss, test_loss=test_loss,
                    train_error1=100 - trainacc, test_error1=100 - testacc)
        results.save()
    #    plt.ion()
    #    fig1 = plt.figure(1)
    #	ax1 = fig1.add_subplot(1,1,1)
    #	ax1.plot(epoch,train_loss,'rs-',epoch,test_loss,'bs-')
    #	plt.ylabel('loss')
    #	fig1.canvas.draw()
    #	fig2 = plt.figure(2)
    #	ax2 = fig2.add_subplot(1,1,1)
    #	ax2.plot(epoch,100-trainacc,'rs-',epoch,100-testacc,'bs-')
    #	plt.ylabel('error %')
    #	fig2.canvas.draw()
