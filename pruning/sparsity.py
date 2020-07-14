# Definition of utility functions for sparsity - plotting, calculating metrics, validating

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
torch.set_printoptions(threshold=10000)

import pdb
import src.config as cfg

def sparsity_plot (s_tuple, path="./", save=True):
    """ Plots layer-wise histogram of sparsity - matrix-level, xbar-level

    Arguments:
        s_tuple:
            - dict -- dictionary of layer-wise sparsity
            - dict -- dictionary of matrix-column wise sparsity
            - dict -- dictionary of xbar-column wise sparsity

    Returns:
        None
    """

    scale = 2.0
    fig = plt.figure(figsize=(8.0*scale, 5.0*scale))
    fig.suptitle("Distribution of sparsity across layers") # set global title for all sub-plots
    n = len(s_tuple[1].keys())

    # Use stride to plot selected layers
    stride = 2
    n1 = math.floor(math.sqrt(n // stride))

    i,j = 0,0
    for key in s_tuple[1].keys(): # keys are same for matrix-column level, and xbar-column level sparsity
        i += 1
        if (i%stride != 0):
            continue
        j += 1
        if (j > n1**2):
            break
        temp = fig.add_subplot(n1, n1, j)
        temp.hist(s_tuple[1][key].numpy(), bins=10, density=True, label='matrix', histtype='bar')
        temp.hist(s_tuple[2][key].numpy(), bins=10, density=True, label='xbar', histtype='bar')
        #temp.set_xlabel('Sparsity')
        #temp.set_ylabel('Dist')
        temp.legend(prop={'size': 10})
        temp.set_title(key)
    
    #plt.tight_layout()
    if save:
        assert(path[-1] == '/'), "path must end with /"
        filepath = path + 'sparsity_distribution'
        plt.savefig(filepath)
    else:
        plt.show()


def sparsity_metrics (model):
    """ Compute sparsity etimates at different levels.
    
    Levels: layer-level, matrix-column level, 
    xbar-column level.

    Arguments:
        model {torch.nn.Module}

    Returns:
        tuple:
            - int -- layer-level sparsity
            - int -- matrix column-level sparsity
            - int -- xbar column-level sparsity

    Remarks:
        For dataParallel models, pass model.module as arguments #TODO
    """

    layer_s, matrix_s, xbar_s = {}, {}, {}

    for name, module in model.named_modules():

        # TODO: Add for Linear layer
        
        if isinstance(module, torch.nn.Conv2d):
            weight = module.weight            
            
            ## layer-level sparsity
            layer_sparsity = float(torch.sum(weight == 0)) / float(weight.nelement())

            ## convert weight tensor to 2d matrix
            weight_channels_out = weight.shape[0]
            weight_channels_in = weight.shape[1]
            weight_row = weight.shape[2]
            weight_col = weight.shape[3]
            length = weight_channels_in * weight_row * weight_col
            weight_mat2d = weight.view(weight_channels_out, length).t()

            ## Convert 2d matrix to xbars, including padding boundary tiles with zeros
            xbar_row = math.ceil(weight_mat2d.shape[0]/cfg.xbar_row_size)
            xbar_col = math.ceil(weight_mat2d.shape[1]/cfg.xbar_col_size)
            weight_xbar = torch.zeros(xbar_row*cfg.xbar_row_size, xbar_col*cfg.xbar_col_size)
            weight_xbar[:weight_mat2d.shape[0], :weight_mat2d.shape[1]] = weight_mat2d
            xbars = weight_xbar.unfold(0,cfg.xbar_row_size, cfg.xbar_row_size).unfold(1, cfg.xbar_col_size, cfg.xbar_col_size)
            assert (xbar_row == xbars.shape[0] and xbar_col == xbars.shape[1]), "xbars unfolding is incorrect"
            
            ## matrix-level sparsity
            weight_xbar_z = torch.zeros(weight_xbar.shape) # view creates references i.e. no memory replication
            weight_xbar_nz = torch.ones(weight_xbar.shape)
            matrix_sparsity = torch.where(weight_xbar==0.0, weight_xbar_nz, weight_xbar_z)
            matrix_sparsity = torch.sum(matrix_sparsity, 0) / weight_xbar.shape[0] #reduce acorss rows of the matrix

            ## xbar-level sparsity
            xbars_z = torch.zeros(xbars.shape)
            xbars_nz = torch.ones(xbars.shape)
            xbars_sparsity = torch.where(xbars==0.0, xbars_nz, xbars_z)
            xbars_sparsity = torch.sum(xbars_sparsity, 2)/ xbars.shape[2] #reduce across rows in one crossbar (xbar_row_size)

            ## collect stats
            layer_s.update({name: layer_sparsity})
            matrix_s.update({name: matrix_sparsity})
            xbar_s.update({name: xbars_sparsity.view(-1)})

    return layer_s, matrix_s, xbar_s

def sparsity_validate (model):
    """ Prints the global, and layer-wise sparsity of model for validation

    Arguments:
        model {torch.nn.Module}
    
    Returns:
        None
    """

    global_zeros,global_elements = 0,0

    for name, module in model.named_modules():
        local_zeros,local_elements = 0,0
        if isinstance(module, torch.nn.Conv2d):
            local_zeros = float(torch.sum(module.weight==0.0))
            local_elements = float(module.weight.nelement())
            print("Sparsity in " + name + ": {:.2f}%" .format(100*local_zeros/local_elements))

            global_zeros += local_zeros
            global_elements += local_elements
    print("Global sparsity: {:.2f}%" .format(100*global_zeros/global_elements))
