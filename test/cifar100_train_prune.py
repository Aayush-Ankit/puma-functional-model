### Script to train/retrain models with pruning 

import os
import sys

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
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import torch.nn.utils.prune as prune
import pruning.prune as prune_custom # custom pruning

#torch.set_default_tensor_type(torch.HalfTensor)

# User-defined packages
import models
from utils.data import get_dataset
from utils.preprocess import get_transform
from utils.utils import *
from pruning.sparsity import *
import src.config as cfg

if cfg.if_bit_slicing:
    from src.pytorch_mvm_class_v3 import *
else:
    from src.pytorch_mvm_class_no_bitslice import *

# Set seeds in torch, numpy, cudnn, workers to make deterministic execution
torch_seed = 0
torch.manual_seed(torch_seed)
torch.cuda.manual_seed_all(torch_seed)
np.random.seed(torch_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(torch_seed)
os.environ['PYTHONHASHSEED'] = str(torch_seed)

def _init_fn(worker_id):
    np.random.seed(torch_seed)
    random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(torch_seed)

# Find model names/archs that can be run with this script
model_names = []
for path, dirs, files in os.walk(models_dir):
    for file_n in files:
        if (not file_n.startswith("__")):
            model_names.append(file_n.split('.')[0])
    break # only traverse top level directory
model_names.sort()

# Runs one epoch of training on a model
def train(epoch):
    print ("Training...")
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, target) in enumerate(trainloader):
        t_start = time.time()
        
        data, target = inputs.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        t_end = time.time()

        if (not args.evaluate):
            writer.add_scalars('Train', {'top1':top1.avg, 'top5':top5.avg, 'loss':losses.avg}, epoch*len(trainloader)+batch_idx) 
        
        if (batch_idx % args.log_freq == 0 or batch_idx == len(trainloader)-1):
            print('Epoch: [{0}][{1}/{2}({3:.0f}%)]\t'
                  'Time/Batch {4:4.2}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\tLR: {LR}\t' .format(
                   epoch, batch_idx, len(trainloader), 100. *float(batch_idx)/len(trainloader), t_end-t_start,
                   loss=losses, top1=top1, top5=top5, LR = optimizer.param_groups[0]['lr']))
    acc = top1.avg

    #if epoch == 1:
    #print('Train Epoch: {}\t({:.2f}%)]\tLoss: {:.6f}\n'.format(
    #            epoch, acc, losses.avg))
    return acc, losses.avg

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
        
        if (not args.evaluate):
            writer.add_scalars('Validation', {'top1':top1.avg, 'top5':top5.avg, 'loss':losses.avg}, epoch*len(trainloader)+batch_idx) 
        
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
    parser.add_argument('--lr', action='store', type=float, default='1e-4',
                help='the intial learning rate')
    parser.add_argument('--momentum', action='store', type=float, default='0.9',
                help='momentum')
    parser.add_argument('--decay', action='store', type=float, default='0.0005',
                help='weight decay')
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                help='number of total epochs to run')
    parser.add_argument('--evaluate', action='store_true',
                help='evaluate the model')

    # model/dataset etc
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                help='dataset name or folder')
    parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20', choices=model_names,
                help='name of the model')
    parser.add_argument('--pretrained', action='store', default=None,
                help='the path to the pretrained model')
    
    # result collection
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='../../results',
                help='results dir')
    #parser.add_argument('--save', metavar='SAVE', default=None,
    #            help='experiment name to be used for creating folder within results_dir')
    parser.add_argument('--log_freq', metavar='LOG', type=int, default=100,
                help='frequency of loggin the result in terms of number of batches')
    parser.add_argument('--chpt_freq', metavar='checkpoint', type=int, default=40,
                help='frequency of checkpointing the model in terms of number of epochs')
    
    # others
    parser.add_argument('--input_size', type=int, default=None,
                help='image input size')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='J',
                help='number of data loading workers (default: 8)')
    parser.add_argument('--gpus', default='0', 
                help='gpu ids to be used for dataparallel (default: 0)')
    
    # new features
    parser.add_argument('--mvm', action='store_true', default=None,
                help='if running functional simulator backend')
    parser.add_argument('--prunefrac', type=float, default=0.5, 
                help='pruning fraction to be applied to all layers')
    parser.add_argument('--strategy', action='store', default='local', 
                help='pruning strategy adopted', choices=['local', 'global', 'xbar-static', 'xbar-dynamic'])
    
    # Dump simulation argumemts (command line and functional simulator config)
    args = parser.parse_args()
    print('==> Options:',args)
    if (args.mvm):
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
    weights_conv = []
    weights_lin = []
    bn_data = []
    bn_bias = []
    running_mean = []
    running_var = []
    num_batches = []

    best_acc, best_train_acc = 0, 0

    if not args.pretrained:
        assert (args.evaluate == False), 'Provide a trained model path for evaluation'
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                weights_conv.append(m.weight.data.clone())
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
        if (args.prunefrac == 0.0):
            best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                weights_conv.append(m.weight.data.clone())
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_data.append(m.weight.data.clone())
                bn_bias.append(m.bias.data.clone())
                running_mean.append(m.running_mean.data.clone())
                running_var.append(m.running_var.data.clone())
                num_batches.append(m.num_batches_tracked.clone())
            elif isinstance(m, nn.Linear):
                weights_lin.append(m.weight.data.clone())

    i=j=k=0
    for m in model_mvm.modules():
        if isinstance(m, (Conv2d_mvm, nn.Conv2d)):
            m.weight.data = weights_conv[i]
            i = i+1
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data = bn_data[j]
            m.bias.data = bn_bias[j]
            m.running_mean.data = running_mean[j]
            m.running_var.data = running_var[j]
            m.num_batches_tracked = num_batches[j]
            j = j+1
        #elif isinstance(m, Linear_mvm): #TODO
        elif isinstance(m, nn.Linear):
            m.weight.data = weights_lin[k]
            k=k+1

    # Move required model to GPU (if applicable)
    if args.mvm:
        model = model_mvm
        
    model.to(device)#.half() # uncomment for FP16
    model = torch.nn.DataParallel(model)
    
    # Setup dataset - transformation, dataloader 
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

    # Setup creiterion and optimizer
    criterion = nn.CrossEntropyLoss()
    params = model.parameters()
    #optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.SGD(params, momentum=args.momentum, lr=args.lr, weight_decay=args.decay, nesterov=True,dampening=0)
    
    # Create directory to store tensorboard logs
    if (args.strategy in ["global", "local", "xbar-static"]):
        exp_name = args.dataset + "-" + args.model + "-pf-" + str("{:0.2f}" .format(args.prunefrac))
    else: # exp_name of xbar-dynamic is determined by the pretrained model and prunefrac
        pretrained_model_name = args.pretrained.split("/")[-2]
        exp_name = args.dataset + "-" + args.model + "-pf-" + str("{:0.2f}" .format(args.prunefrac)) + "-pre-" + pretrained_model_name
    save_path = os.path.join(args.results_dir, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if (not args.evaluate): # Log simulation only for training/retraining
        writer = SummaryWriter(save_path)
    
    # Prune original model (i.e. not model_mvm)
    if (args.strategy == 'local'):
        for name, module in model.module.named_modules(): # added module for dataParallel
            if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=args.prunefrac) # (use 0.2 to prune 20% weights)
    elif (args.strategy == 'global'):
        parameters_to_prune = []
        for name, module in model.module.named_modules():
            if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
                parameters_to_prune.append ((module, 'weight'))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=args.prunefrac)
    elif ('xbar' in  args.strategy):
        for name, module in model.module.named_modules(): # added module for dataParallel
            if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
                # For unpruned networks
                if (args.strategy == 'xbar-static'):
                    # Here: threshold is interpreted as prune amount at xbar column level
                    prune_custom.l1_xbar_unstructured(module, name='weight', threshold=args.prunefrac, xbar_strategy='static') # (use 0.5 on unpruned network to have all xbar columns to be 50% sparse)
                # For previously pruned networks
                elif (args.strategy == 'xbar-dynamic'):
                    # Here: threshold is interpreted as prune threshold for xbar col outlier rejection
                    prune_custom.l1_xbar_unstructured(module, name='weight', threshold=args.prunefrac, xbar_strategy='dynamic')
                else:
                    assert (0), "specified xbar pruning type not unsupported"
    else:
        assert(0), "specified pruring strategy not supported"

    sparsity_validate (model) # validate sparsity after pruning
    # print(dict(model.named_buffers()).keys())  # verify that all masks exist (masks are added as register_buffers)
    
    if (args.evaluate):
        [test_acc, test_loss] = test()
        print ("Testing accuracy: ", test_acc)
    else:
        for epoch in range(0, args.epochs):
            # adjust_learning_rate(optimizer, epoch)
            [train_acc, train_loss] = train(epoch)
            [test_acc, test_loss] = test()
            
            # checkpoint model (best of regular)
            state = {
                'epoch': epoch, 
                'best_acc': best_acc, 
                'test_acc': test_acc,
                'state_dict': model.module.state_dict(), # module for dataParallel
                'optimizer_state_dict': optimizer.state_dict()
            }
            
            isbest = test_acc > best_acc # best model needs to be checkpointed
            if (isbest):
                best_acc = test_acc
                print ("Best accuracy: ", best_acc)

            isregular = (epoch % args.chpt_freq == 0) # model needs to be checkpointed regularly
            print("==> Saving for regular checkpointing")
            # remove pruning
            for name, module in model.module.named_modules(): # added module for dataParallel
                if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
                    prune.remove(module, name='weight')
            state['state_dict'] = model.module.state_dict()
            save_checkpoint(state, isbest, save_path, 'checkpoint.pth.tar', isregular)
            # restore pruning
            for name, module in model.module.named_modules(): # added module for dataParallel
                if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
                    prune.l1_unstructured(module, name='weight', amount=args.prunefrac)
    
    exit(0)