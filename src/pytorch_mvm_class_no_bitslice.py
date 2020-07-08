import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn import init
import math
import numpy as np
import pdb
import time
import sys
torch.set_printoptions(threshold=10000)

import config as cfg

class Conv2d_mvm_function(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # +--------------------------+
    # |            MVM           |   
    # +--------------------------+
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, tile_row=1, tile_col=1, xbar_row_size=64, xbar_col_size=32):
       
        #torch.set_default_tensor_type(torch.HalfTensor) # uncomment for FP16
        
        ## fixed-16: 
        ## sign     : 1 
        ## integer  : 3
        ## fraction : 12
        if weight_bit_frac == -1:
            weight_bit_frac = weight_bits//4*3
        if input_bit_frac == -1:
            input_bit_frac = input_bits//4*3

        device = input.device

        ## Format weights: reshape weight tensor to 2d matrix and then to xbars
        ## xbars is a tiled representation of 2d matrix
        # Convert weight tensor to 2d matrix
        weight_channels_out = weight.shape[0]
        weight_channels_in = weight.shape[1]
        weight_row = weight.shape[2]
        weight_col = weight.shape[3]
        length = weight_channels_in * weight_row * weight_col
        weight_mat2d = weight.view(weight_channels_out, length).t()
        
        # Covert 2d matrix to xbars, including padding boundary tiles with zeros
        xbar_row = math.ceil(weight_mat2d.shape[0]/cfg.xbar_row_size)
        xbar_col = math.ceil(weight_mat2d.shape[1]/cfg.xbar_col_size)
        weight_xbar = torch.zeros(xbar_row*cfg.xbar_row_size, xbar_col*cfg.xbar_col_size).to(device)
        weight_xbar[:weight_mat2d.shape[0], :weight_mat2d.shape[1]] = weight_mat2d
        xbars = weight_xbar.unfold(0,cfg.xbar_row_size, cfg.xbar_row_size).unfold(1, cfg.xbar_col_size, cfg.xbar_col_size)
        assert (xbar_row == xbars.shape[0] and xbar_col == xbars.shape[1]), "xbars unfolding is incorrect"
        
        ## Format inputs: accomodate for padding/stride and then reshape to xbars
        input_batch = input.shape[0]
        input_channels = input.shape[1]     # weight_channels_in == input_channels
        input_row = input.shape[2] + padding[0]*2
        input_col = input.shape[3] + padding[1]*2
        input_pad = torch.zeros(input_batch, input_channels, input_row, input_col).to(device)
        input_pad[:,:,padding[0]:input_row-padding[0],padding[1]:input_col-padding[1]] = input
        
        ## Format outputs: accomodate for padding/stride and then reshape to xbars
        output_row = (input_row - weight_row)//stride[0] + 1
        output_col = (input_col - weight_col)//stride[1] + 1 
        output = torch.zeros(input_batch, weight_channels_out, output_row, output_col).to(device)
        
        ## Tiled For-loop: output feature map size should be multiple of tile size
        if (tile_row > output_row):
            tile_row = output_row
        if (tile_col > output_col):
            tile_col = output_col
        assert output_row%tile_row == 0 and output_col%tile_col == 0, "Output feature map size should be multiple of tile size"
        
        num_pixel = tile_row*tile_col
        flatten_input = torch.zeros(input_batch*num_pixel, xbar_row*cfg.xbar_row_size).to(device)
        unfold = nn.Unfold(kernel_size=(weight_row, weight_col), stride=(stride[0], stride[1]))
                
        input_patch_row = (tile_row-1)*stride[0] + weight_row
        stride_input_row = stride[0]*tile_row
        input_patch_col = (tile_col-1)*stride[1] + weight_col
        stride_input_col = stride[1]*tile_col
        
        for i in range(math.ceil(output_row/tile_row)):
            for j in range(math.ceil(output_col/tile_col)):
                batch_size_t = input_batch*num_pixel # effective batch size with tiling
                input_temp = unfold(input_pad[:,:, stride_input_row*i:stride_input_row*i+input_patch_row, stride_input_col*j:stride_input_col*j+input_patch_col]).permute(2,0,1) # dimensions after unfold+permute: patches, batchsize, k^2*I
                input_temp = input_temp.reshape(batch_size_t,-1) # dimensions: batch_size*#_of_output_pixel, :    
                
                flatten_input[:,:input_temp.shape[1]] = input_temp
                flatten_input_xbar = flatten_input.reshape(batch_size_t, xbar_row, cfg.xbar_row_size)

                # reshape input to conform to broadcast for torch.mul
                xbars_in = flatten_input_xbar.reshape(batch_size_t, xbar_row, 1, xbar_row_size, 1)

                # matrix-vector multiplication
                xbars_out = torch.mul(xbars, xbars_in)
                xbars_out = torch.sum(torch.sum(xbars_out,3),1) # reduce within a xbar (xbar_row_size), and then across xbars (xbar_row)
                xbars_out = xbars_out.reshape(batch_size_t, -1)

                output[:,:,i*tile_row:(i+1)*tile_row,j*tile_col:(j+1)*tile_col] = xbars_out.reshape(tile_row, tile_col, input_batch, -1).permute(2,3,0,1)[:,:weight_channels_out,:,:]  ## #batchsize, # o/p channels, tile_row, tile_col 
                
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups) 
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)
            
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None 


class _ConvNd_mvm(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'bit_slice', 'bit_stream','weight_bits', 'weight_bit_frac','input_bits', 'input_bit_frac']

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, check_grad=False, 
                 weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, tile_row=1, tile_col=1, xbar_row_size=64, xbar_col_size=32):
        super(_ConvNd_mvm, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        # Functional simulator parameters
        self.weight_bits = cfg.weight_bits if cfg.ifglobal_weight_bits else weight_bits
        self.weight_bit_frac = cfg.weight_bit_frac if cfg.ifglobal_weight_bit_frac else weight_bit_frac
        self.input_bits = cfg.input_bits if cfg.ifglobal_input_bits else input_bits
        self.input_bit_frac = cfg.input_bit_frac if cfg.ifglobal_input_bit_frac else input_bit_frac
        self.tile_col = cfg.tile_col if cfg.ifglobal_tile_col else tile_col
        self.tile_row = cfg.tile_row if cfg.ifglobal_tile_row else tile_row
        self.xbar_col_size = cfg. xbar_col_size if cfg.ifglobal_xbar_col_size else xbar_col_size
        self.xbar_row_size = cfg. xbar_row_size if cfg.ifglobal_xbar_row_size else xbar_row_size

        if check_grad:
            tensor_constructor = torch.DoubleTensor # double precision required to check grad
        else:
            tensor_constructor = torch.Tensor # In PyTorch torch.Tensor is alias torch.FloatTensor

        if transposed:
            self.weight = nn.Parameter(tensor_constructor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(tensor_constructor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2d_mvm(_ConvNd_mvm):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', check_grad=False,
                 weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, tile_row=1, tile_col=1, xbar_row_size=64, xbar_col_size=32):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(Conv2d_mvm, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode,
            weight_bits, weight_bit_frac, input_bits, input_bit_frac, tile_row, tile_col)
    #@weak_script_method
    def forward(self, input):
            return Conv2d_mvm_function.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
            self.weight_bits, self.weight_bit_frac, self.input_bits, self.input_bit_frac, self.tile_row, self.tile_col, self.xbar_row_size, self.xbar_col_size)
