# Definition of custom pruning function - crossbar-aware unstructured sparsity

import math
import sys
import pdb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.utils.prune as prune
torch.set_printoptions(threshold=10000)

import src.config as cfg

class L1XbarUnstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones
    with the lowest L1-norm such that:
        1) pruning is done on a per xbar column basis to reduce adc resolution
        2) xbar column sparsity reaches one of 0.50, 0.75, 0.88, 0.94 
    
    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    # TODO - implement outlier rejection
    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(
                torch.abs(t).view(-1), k=nparams_toprune, largest=False
            )
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """
        return super(L1XbarUnstructured, cls).apply(module, name, amount=amount)


def l1_xbar_unstructured(module, name, amount):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified `amount` of (currently unpruned) units with the
    lowest L1-norm such that:
    1) pruning is done on a per xbar column basis to reduce adc resolution
    2) xbar column sparsity reaches one of 0.50, 0.75, 0.88, 0.94 
    
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
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module
    Examples:
        >>> m = prune.l1_xbar_unstructured(nn.Linear(2, 3), 'weight', amount=0.2)
        >>> m.state_dict().keys()
        odict_keys(['bias', 'weight_orig', 'weight_mask'])
    """
    L1XbarUnstructured.apply(module, name, amount)
    return module
