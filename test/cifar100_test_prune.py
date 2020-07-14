### Script to test pruned models for accuracy and sparsity

import os
import sys
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(root_dir, "models")
datasets_dir = os.path.join(root_dir, "datasets")
src_dir = os.path.join(root_dir, "src")
#test_dir = os.path.join(root_dir, "test")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
sys.path.insert(0, models_dir)
sys.path.insert(0, datasets_dir)
sys.path.insert(0, src_dir)
#sys.path.insert(0, test_dir) 

# Standard or Built-in packages
import numpy as np
import random
import argparse
import pdb

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

#torch.set_default_tensor_type(torch.HalfTensor)

# User-defined packages
import models
from utils.data import get_dataset
from utils.preprocess import get_transform
from utils.utils import *
from pruning.sparsity import *

import src.config as cfg

# Find model names/archs that can be run with this script
model_names = []
for path, dirs, files in os.walk(models_dir):
    for file_n in files:
        if (not file_n.startswith("__")):
            model_names.append(file_n.split('.')[0])
    break # only traverse top level directory
model_names.sort()

# Run evaluation on a model (<model>.py)
def test():
    print ("Testing...")
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx,(data, target) in enumerate(testloader):
        t_start = time.time()
        
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss= criterion(output, target)
        
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        losses.update(loss.data, data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        t_end = time.time()
        
        if (batch_idx % args.log_freq == 0 or batch_idx == len(testloader)-1):
            print('[{0}/{1}({2:.0f}%)]\t'
                  'Time/Batch {3:4.2f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   batch_idx, len(testloader), 100. *float(batch_idx)/len(testloader), t_end-t_start,
                   loss=losses, top1=top1, top5=top5))
            #if batch_idx == 10:
            #    break
    
    #print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #      .format(top1=top1, top5=top5))
    acc = top1.avg
    return acc, losses.avg

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # training/test setup
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N', 
                help='mini-batch size (default: 256)')

    # model/dataset etc
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                help='dataset name or folder')
    parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20', choices=model_names,
                help='name of the model')
    parser.add_argument('--pretrained', action='store', default=None,
                help='the path to the pretrained model')
    
    # result collection
    parser.add_argument('--log_freq', metavar='LOG', type=int, default=100,
                help='frequency of loggin the result in terms of number of batches')
    
    # others
    parser.add_argument('--input_size', type=int, default=None,
                help='image input size')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='J',
                help='number of data loading workers (default: 8)')
    parser.add_argument('--gpus', default='0', 
                help='gpu ids to be used for dataparallel (default: 0)')
    
    # new features
    
    # Dump simulation argumemts (command line and functional simulator config)
    args = parser.parse_args()
    print('==> Options:',args)
    cfg.dump_config()

    os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device available:', device)
    print('GPU Id(s) being used:', args.gpus)

    print('==> Building model and model_mvm for', args.model, '...')
    if (args.model in model_names and args.model+'_mvm' in model_names):
        model = (__import__(args.model)) #import module using the string/variable_name
        model_mvm = (__import__(args.model+'_mvm'))
    else:
        raise Exception(args.model+'is currently not supported')
        
    # Extract the function capturing model definition
    model = model.net()
    model_mvm = model_mvm.net(cfg.non_ideality)
    #print(model)

    # Load parameters (to model and model_mvm) from a pretrained model
    print('==> Initializing model parameters ...')
    best_acc, best_train_acc = 0, 0

    if not args.pretrained:
        assert (0), 'Provide a trained model path for evaluation'
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        # best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])
    
    # Setup dataset - transformation, dataloader 
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)

    test_data = get_dataset(args.dataset, 'val', transform['eval'])
    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    
    model.to(device)#.half() # uncomment for FP16
    #model = torch.nn.DataParallel(model)
    
    [test_acc, test_loss] = test()
    print ("Testing accuracy: ", test_acc)

    result = sparsity_metrics(model)

    # save the sparsity plot in same path as pretrained model
    temp_l = args.pretrained.split("/")[0:-1] # remove .tar file from path
    filepath = ''
    for t in temp_l:
        filepath += (t+"/")
    sparsity_plot(result, path=filepath)
    
    exit(0)