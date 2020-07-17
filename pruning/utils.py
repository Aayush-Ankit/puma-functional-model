## Defines utility files to add/remove pruning from a model (use model.module for dataParallel models)

import torch
import torch.nn.utils.prune as prune
import pruning.prune as prune_custom # custom pruning

# Add pruning based on specified strategy
def add_pruning(model, strategy, prunefrac):
    if (strategy == 'local'):
        for name, module in model.module.named_modules(): # added module for dataParallel
            if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=prunefrac) # (use 0.2 to prune 20% weights)
    elif (strategy == 'global'):
        parameters_to_prune = []
        for name, module in model.module.named_modules():
            if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
                parameters_to_prune.append ((module, 'weight'))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prunefrac)
    elif ('xbar' in  strategy):
        for name, module in model.module.named_modules(): # added module for dataParallel
            if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
                # For unpruned networks
                if (strategy == 'xbar-static'):
                    # Here: threshold is interpreted as prune amount at xbar column level
                    prune_custom.l1_xbar_unstructured(module, name='weight', threshold=prunefrac, xbar_strategy='static') # (use 0.5 on unpruned network to have all xbar columns to be 50% sparse)
                # For previously pruned networks
                elif (strategy == 'xbar-dynamic'):
                    # Here: threshold is interpreted as prune threshold for xbar col outlier rejection
                    prune_custom.l1_xbar_unstructured(module, name='weight', threshold=prunefrac, xbar_strategy='dynamic')
                else:
                    assert (0), "specified xbar pruning type not unsupported"
    else:
        assert(0), "specified pruring strategy not supported"


# Remove pruning
def remove_pruning(model):
    for name, module in model.module.named_modules(): # added module for dataParallel
        if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
            prune.remove(module, name='weight')

