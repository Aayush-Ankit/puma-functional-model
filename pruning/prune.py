## Definition of custom pruning function - crossbar-aware unstructured sparsity

import math
import torch
import torch.nn.utils.prune as prune

## Uncomment (lines 9-27) for debugging (running unit test i.e. __main__) set seed for debugging/reproducibility - if debugging with values

#import os
#import sys
#root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH

#import random
#import numpy as np
#import torch.nn as nn
#import torch.nn.functional as F
#torch.set_printoptions(threshold=10000)

#torch_seed = 0
#torch.manual_seed(torch_seed)
#torch.cuda.manual_seed_all(torch_seed)
#np.random.seed(torch_seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#random.seed(torch_seed)
#os.environ['PYTHONHASHSEED'] = str(torch_seed)

import src.config as cfg

class L1XbarUnstructured(prune.BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones
    with the lowest L1-norm such that:
        1) pruning is done on a per xbar column basis to reduce adc resolution
            a) static - makes sparsity on xbar_column basis to specified amount (or threshold)
                - This strategy is for unpruned network i.e. making unstructured sparsity on xbar column basis (alike layer or global unstructured)
            b) dynamic - reduce the exisitng sparsity on xbar_column to one of 0.50, 0.75, 0.88, 0.94, 0.97
                - This strategy is for previously pruned networks
    """

    PRUNING_TYPE = "unstructured"
    
    def __init__(self, threshold, xbar_strategy='dynamic'):
        self.threshold = threshold
        self.xbar_strategy = xbar_strategy

    def compute_mask(self, t, default_mask):
        """ Perform outlier rejection to bring xbar columns within threshold (th)
        to one of 0%, 50%, 75%, 88%, 93% sparsity for 8b, 7b, 6b, 5b, 4b ADC.
        
        Note: This class is intended to act on an previously pruned module.
        Col sparsity mapping (before -> after), with baseline 8b ADC:
            - 50-x to 50 -> 50 // 7b ADC
            - 75-x to 75 -> 75 // 6b ADC
            - 88-x to 88 -> 88 // 5b ADC
            - 93-x to 93 -> 93 // 4b ADC

        Arguments:
            threshold: float
            xbar_strategy: string
        Returns:
            mask: mask (tensor)
        """ 

        # receives a column tensor and calculates sparsity when mapped to an xbar column
        def _get_sparsity (col_t):
            n = col_t.nelement()
            extra_zeros = 0 if n == cfg.xbar_row_size else cfg.xbar_row_size - n
            return float(torch.sum(col_t == 0.0) + extra_zeros) / float(cfg.xbar_row_size), extra_zeros

        # receives a column tensor and implements logic to give number of params to prune
        def _get_n_params_to_prune (col_t, threshold, xbar_strategy):
            assert (xbar_strategy in ['static', 'dynamic']), "Unsupported xbar pruning strategy"
            
            sparsity, extra_zeros = _get_sparsity(col_t)
            if (xbar_strategy == 'dynamic'):
                sparsity_out = sparsity
                if (sparsity >= 0.5-threshold*(0.5-0.0) and sparsity <= 0.5):
                    sparsity_out = 0.5
                elif (sparsity >= 0.75-threshold*(0.75-0.5) and sparsity <= 0.75):
                    sparsity_out = 0.75
                elif (sparsity >= 0.88-threshold*(0.88-0.75) and sparsity <= 0.88):
                    sparsity_out = 0.88
                elif (sparsity >= 0.94-threshold*(0.94-0.88) and sparsity <= 0.94):
                    sparsity_out = 0.94
                elif (sparsity >= 0.97-threshold*(0.97-0.94) and sparsity <= 0.97):
                    sparsity_out = 0.97
            else:
                sparsity_out = threshold

            temp = int((sparsity_out*cfg.xbar_row_size))
            if (temp < extra_zeros): # more zeros than required to start with
                n_params_to_prune = 0
            else:
                n_params_to_prune = temp - extra_zeros
            assert (n_params_to_prune >= 0 and n_params_to_prune<=col_t.nelement()), "Logic for number of parameters to prune is incorrect"
            
            return n_params_to_prune

        # Modify mask for every xbar_column based on sparsity logic
        def _get_xbar_mask (weight_mat, num_row, num_col):
            weight_mat_mask = (weight_mat != 0.0).float()
            for i in range(num_row):
                for j in range(math.ceil(num_col/cfg.xbar_row_size)):
                    col_start = j*cfg.xbar_row_size
                    col_end = (j+1)*cfg.xbar_row_size
                    if (col_end > num_col):
                        col_end = num_col

                    xbar_col_wt = weight_mat[i, col_start:col_end]
                    n_params_to_prune = _get_n_params_to_prune(xbar_col_wt, self.threshold, self.xbar_strategy)

                    # Fix GPU Bugs - skip for n_params_to_prune = 0
                    if (n_params_to_prune < 1):
                        continue

                    topk = torch.topk(
                        torch.abs(xbar_col_wt).view(-1), k=n_params_to_prune, largest=False
                    )
                    weight_mat_mask[i, col_start:col_end].view(-1)[topk.indices] = 0.0

            return weight_mat_mask
        
        
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        weight = t
        # Crossbar aware pruning is only applicable for nn.Linear and nn.Conv2d layers' weight tensor
        assert (len(t.shape) in [2,4]), "Unsupported layer type for L1XbarUnstructured pruning"

        if (len(t.shape) == 4): #torch.nn.Conv2d
            # Convert weight tensor to 2d matrix
            weight_channels_out = weight.shape[0]
            weight_channels_in = weight.shape[1]
            weight_row = weight.shape[2]
            weight_col = weight.shape[3]
            length = weight_channels_in * weight_row * weight_col
            weight_mat = weight.view(weight_channels_out, length) # dimensions (out_channels, in_channels*k*k)
            
            mask = _get_xbar_mask(weight_mat, weight_channels_out, length)
            
            # Reshape mask to weight's dimensions
            mask = mask.view(-1, weight_channels_in, weight_row, weight_col)

        elif (len(t.shape) == 2): # torch.nn.Linear
            # Convert weight tensor to 2d matrix
            weight_channels_out = weight.shape[0]
            weight_channels_in = weight.shape[1]
            
            mask = _get_xbar_mask(weight, weight_channels_out, weight_channels_in)

        return mask


    @classmethod
    def apply(cls, module, name, threshold, xbar_strategy):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            threshold (float): sparsity threshold to reduce adc precision
                - should be between 0.0 and 1.0 and represent the
            xbar_strategy (string): strategy for xbar pruning

        """
        return super(L1XbarUnstructured, cls).apply(module, name, threshold=threshold, xbar_strategy=xbar_strategy)


def l1_xbar_unstructured(module, name, threshold, xbar_strategy):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing units with the
    lowest L1-norm such that:
    1) pruning is done on a per xbar column basis to reduce adc resolution
    2) xbar column sparsity reaches one of 0.50, 0.75, 0.88, 0.94, 0.97 depending on threshold
    
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called ``name+'_mask'`` corresponding to the
    binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    ``name+'_orig'``.
    
    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        threshold (float): specified threshold used for reducing ADC precision
        xbar_strategy (string): specified xbar pruning strategy - 'static', 'dynamic'
    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module
    Examples:
        >>> m = prune.l1_xbar_unstructured(nn.Linear(2, 3), 'weight')
        >>> m.state_dict().keys()
        odict_keys(['bias', 'weight_orig', 'weight_mask'])
    """
    L1XbarUnstructured.apply(module, name, threshold, xbar_strategy)
    return module


## Unit test for l1_xbar_unstructured pruning
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    ## Stress test with varying values of C, K, R and cfg.xbar_row_size
    def stress_test ():
        while(True):
            C = random.randint(1, 100) % 65 + 3
            K = random.randint(1, 100) % 65 + 16
            R = random.randint(1, 100) % 8 + 1
            cfg.xbar_row_size = random.randint(1, 100) % 64 + 1
            xbar_strategy  = random.randint(0,1)
            xbar_strategy_name  = {0:"dynamic", 1:"static"}
            #C, K, R = 3, 3, 1
            #cfg.xbar_row_size = 1

            class LeNet(nn.Module):
                def __init__(self):
                    super(LeNet, self).__init__()
                    #self.conv1 = nn.Conv2d(C, K, R)
                    self.fc1 = nn.Linear(C, K)

                def forward(self, x):
                    #x = self.conv1(x)
                    x = self.fc1(x)
                    return x

            model = LeNet().to(device=device)
    
            def sparsity(weight):
                return float(torch.sum(weight==0.0))/float(weight.nelement())

            #module = model.conv1
            #module.weight = torch.nn.Parameter(torch.arange(0,C*K*R*R).view(K, C, R, R).float())
            module = model.fc1
            #module.weight = torch.nn.Parameter(torch.arange(0,C*K).view(K, C).float())
    
            try:
                # Prune a model initially - unstructured pruning
                if (xbar_strategy_name[xbar_strategy] == 'dynamic'):
                    prune.l1_unstructured(module, name="weight", amount=0.5)
                    prune.remove(module, 'weight')
                s_mat = sparsity(module.weight)

                # Fine-tune with xbar-aware pruning
                l1_xbar_unstructured(module, name="weight", threshold=0.5, xbar_strategy=xbar_strategy_name[xbar_strategy])
                s_xbar = sparsity(module.weight)
                
                #print("{:.2f}" .format(s_xbar))
                prune.remove(module, 'weight')
                print ('Passed:\t Mat {0:0.2f}\t Xbar {1:0.2f} \t [C {2}, K {3}, R {4}, xbar_row_size {5}] Xbar strategy {6}' 
                .format(s_mat, s_xbar, C, K, R, cfg.xbar_row_size, xbar_strategy_name[xbar_strategy]))
                assert (s_xbar >= s_mat)
            except Exception as e:
                print ("Failed configuration [C, K, R, xbar_row_size, strategy] ", [C, K, R, cfg.xbar_row_size, xbar_strategy_name[xbar_strategy]])
                raise e

    stress_test()
