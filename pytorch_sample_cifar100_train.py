
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import pdb
import models
import random
from pytorch_mvm_class_v2 import *
import os
import argparse
from data import get_dataset
from preprocess import get_transform
from utils import *
from torchvision.utils import save_image
import pdb
torch_seed=0
np.random.seed(torch_seed)
torch.manual_seed(torch_seed)
torch.cuda.manual_seed_all(torch_seed)
os.environ['PYTHONHASHSEED'] = str(torch_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUDA_VISIBLE_DEVICES']= '2'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

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
    torch.save(state, 'models/resnet20fp_cifar10_retrained.pth.tar')
def save_state_batch(model, best_acc, batch_idx):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    filename = 'models/resnet20fp_cifar10_retrained'+str(batch_idx)+'.pth.tar'
    torch.save(state,  filename)
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
def train(epoch, model):
    global best_train_acc
    model.train()
    training = True
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    correct = 0
    # pdb.set_trace()
    for batch_idx, (inputs, target) in enumerate(trainloader):
        # if batch_idx<121:
            # continue
        # pdb.set_trace()
        t1 = time.time()
        data, target = inputs.to(device), target.to(device)
        # pdb.set_trace()
        optimizer.zero_grad()
        # pdb.set_trace()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        
        optimizer.step()
        prec1, prec5 = accuracy(output.data, target, training, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        if batch_idx % 1 == 0:
            t2 = time.time()
            print('Epoch: [{0}][{1}/{2}({3:.0f}%)]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\tLR: {LR}\tTime Elapsed: {t}'.format(
                       epoch, batch_idx, len(trainloader), 100. *float(batch_idx)/len(trainloader),
                       loss=losses, top1=top1, top5=top5, LR = optimizer.param_groups[0]['lr'], t=t2-t1))
        if batch_idx % 20 == 0:
            save_state_batch(model, best_acc, batch_idx)
    acc = top1.avg

    if acc > best_train_acc:
        best_train_acc = acc
    if epoch == 1:
        save_state(model, best_acc)
    print('Train Epoch: {}\t({:.2f}%)]\tLoss: {:.6f}\n'.format(
                epoch,
                acc, losses.avg))

    print('Best Accuracy: {:.2f}%\n'.format(best_train_acc))
    return acc, losses.avg
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
        if batch_idx == 10:
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
        target_var = torch.autograd.Variable(target.cuda(), volatile=True)

                                    
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
        # if batch_idx == 10:
        #     break        

    acc = top1.avg
 
    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return acc, losses.avg
def adjust_learning_rate(optimizer, epoch):
    update_list = [5,10]
    lr_list = [1e-4,1e-5]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            index_epoch = update_list.index(epoch)
            param_group['lr'] = lr_list[index_epoch]
    return


## To Indranil & Mustafa: This is for using 'for loops' in mvm_tensor. Just execute with '-i' at command line
# ind = False
# for i in range(len(sys.argv)):
#     if sys.argv[i] == '-i':
#         ind = True
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                         metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-i', default=False,
                         metavar='N', help='turn on Ind feature')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                help='dataset name or folder')
    parser.add_argument('--lr', action='store', default='1e-4',
        help='the intial learning rate')
    parser.add_argument('--arch', action='store', default='resnet20',
        help='the architecture for the network: resnet')
    parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20',
                choices=model_names,
                help='model architecture: ' +
                ' | '.join(model_names) +
                ' (default: resnet)')
    parser.add_argument('--momentum', action='store', default='0.9',
        help='momentum')
    parser.add_argument('--pretrained', action='store', default=None,
        help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
        help='evaluate the model')
    parser.add_argument('--input_size', type=int, default=None,
                help='image input size')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                help='number of data loading workers (default: 8)')
    parser.add_argument('-cuda', '--cuda_gpu', default=0, type=int, metavar='N',
                help='gpu index (default: 8)')
    parser.add_argument('-exp', '--experiment', default='16x16', metavar='N',
                help='experiment name')
    parser.add_argument('--gpus', default='1',
                help='gpus used for training - e.g 0,1,3')
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                help='number of total epochs to run')
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
    
    if args.i == 'True':
        ind = True
    else:
        ind = False
    #



    print('==> building model',args.arch,'...')
    if args.arch == 'vgg' or 'resnet20':
        #print(models.__dict__)
        model = models.__dict__[args.model]
        #model_config = {'input_size': args.input_size, 'dataset': args.dataset}
        print(model)
    else:
        raise Exception(args.arch+' is currently not supported')


    model = model()
    model_mvm = models.__dict__['resnet20_mvm']
    model_mvm = model_mvm(ind)

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
        best_train_acc=0
        pretrained_model = torch.load(args.pretrained)
        best_acc = 0
        # best_acc = pretrained_model['best_acc']
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
            m.weight.data = weights_lin[k]
            k=k+1
    model.cuda()
    model_mvm.cuda()
    def _init_fn(worker_id):
        np.random.seed(torch_seed)
        random.seed(torch_seed)
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(torch_seed)

    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    train_data = get_dataset(args.dataset, 'train', transform['train'])
    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, worker_init_fn = _init_fn)

    test_data = get_dataset(args.dataset, 'val', transform['eval'])
    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, worker_init_fn = _init_fn)

    classes = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100)
    base_lr = float(args.lr)
    param_dict = dict(model_mvm.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr, 
            'weight_decay':0.0005}]
    #optimizer = optim.Adam(params,lr=0.10,weight_decay=0.00001)
        optimizer = optim.SGD(params, momentum = float(args.momentum), lr=0.10,weight_decay=0.0005, nesterov=True,dampening=0)
    criterion = nn.CrossEntropyLoss()


    if args.evaluate:
       # test()
        test_mvm()
        exit(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(1, args.epochs):
        if epoch <2:
            continue
        adjust_learning_rate(optimizer, epoch)
        [trainacc,train_loss] = train(epoch,model_mvm)
        [testacc,test_loss] = test_mvm()
    #net.to(device)
    # mynet.to(device)
    # inputs = inputs.to(device)

    # result_net = model(inputs)
    # print(result_net[0][0])#[0,:2])

    # result_mynet = model_mvm(inputs)
    # print(result_mynet[0][0])#[0,:2])
#print(torch.norm(result_net-result_mynet))

