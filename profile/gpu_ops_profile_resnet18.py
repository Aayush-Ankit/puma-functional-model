### Script to profile the runtime for different ops in a model on GPU

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

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # training/test setup
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', 
                help='mini-batch size (default: 256)')

    # model/dataset etc
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                help='dataset name or folder')
    parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20', choices=model_names,
                help='name of the model')
    parser.add_argument('--pretrained', action='store', default=None,
                help='the path to the pretrained model')
    
    # others
    parser.add_argument('--input_size', type=int, default=None,
                help='image input size')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='J',
                help='number of data loading workers (default: 8)')
    parser.add_argument('--gpus', default='0', 
                help='gpu ids to be used for dataparallel (default: 0)')
    
    # output file
    parser.add_argument('--output', action='store', default=None,
                help='the path to the log file')
    
    # Dump simulation argumemts (command line and functional simulator config)
    args = parser.parse_args()
    print('==> Options:',args)
    cfg.dump_config()

    os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device available:', device)
    print('GPU Id(s) being used:', args.gpus)

    print('==> Building model for', args.model, '...')
    if (args.model in model_names):
        model = (__import__(args.model)) #import module using the string/variable_name
    else:
        raise Exception(args.model+'is currently not supported')
        
    # Extract the function capturing model definition
    model = model.net()
    #print(model)

    # Load parameters (to model) from a pretrained model
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
    
    model.to(device)
    #model.eval()
    model.half() # uncomment for FP16

    losses = AverageMeter()
    top1 = AverageMeter()

    warmup_iter = 10
    profile_iter = 100

    torch.cuda.synchronize() 
    #with torch.cuda.profiler.profile():
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        # warmup for batch
        i= 0
        for batch_idx,(data, target) in enumerate(testloader):
            i += 1
            data, target = data.to(device), target.to(device)
            #output = model(data)
            output = model(data.half())
            loss= criterion(output, target)
            if (i == 10):
                break
        
        # measure for a batch
        i= 0
        #with torch.autograd.profiler.emit_nvtx() as prof_cuda:
        for batch_idx,(data, target) in enumerate(testloader):
            i += 1
            data, target = data.to(device), target.to(device)
            #output = model(data)
            output = model(data.half())
            loss= criterion(output, target)

            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.data, data.size(0))
            top1.update(prec1[0], data.size(0))
            if (i == 100):
                break

    if (args.output == None):
        print(prof.key_averages())
    else:
        original_stdout = sys.stdout
        with open(args.output, 'w') as f:
            sys.stdout = f
            print(prof.key_averages())
            sys.stdout = original_stdout


    #torch.autograd.profiler.load_nvprof("test_trace1.prof")

    exit(0)