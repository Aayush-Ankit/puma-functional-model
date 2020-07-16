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

import src.config as cfg

from geneix.mvm_dataset import *

class Conv2d_mvm_function(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # +--------------------------+
    # |            MVM           |   
    # +--------------------------+
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, bit_slice=2, bit_stream=1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, tile_row=2, tile_col=2, xbmodel=None, xbmodel_weight_path=None, dataset=False):
       
        #torch.set_default_tensor_type(torch.HalfTensor)
        ## fixed-16: 
        ## sign     : 1 
        ## integer  : 3
        ## fraction : 12
        num_pixel = tile_row*tile_col
        if weight_bit_frac == -1:
            weight_bit_frac = weight_bits//4*3
        if input_bit_frac == -1:
            input_bit_frac = input_bits//4*3
        if acm_bit_frac == -1:
            acm_bit_frac = acm_bits//4*3
        if adc_bit == -1:
            adc_bit = int(math.log2(cfg.xbar_row_size))
            if bit_stream != 1:
                adc_bit += bit_stream
            if bit_slice != 1:
                adc_bit += bit_slice

        device = input.device
        weight_channels_out = weight.shape[0]
        weight_channels_in = weight.shape[1]
        weight_row = weight.shape[2]
        weight_col = weight.shape[3]
        length = weight_channels_in * weight_row * weight_col
        flatten_weight = torch.zeros(2, weight_channels_out, length).to(device)     ## W+ / W-
    
        weight_temp = weight.reshape((weight_channels_out, length))
        flatten_weight[0] = torch.clamp(weight_temp, min=0)  ## flatten weights
        flatten_weight[1] = torch.clamp(weight_temp, max=0).abs()
        pos_bit_slice_weight = bit_slicing(flatten_weight[0], weight_bit_frac, bit_slice, weight_bits).to(device) ## v2: flatten weights --> fixed point --> bit slice -- v1
        neg_bit_slice_weight = bit_slicing(flatten_weight[1], weight_bit_frac, bit_slice, weight_bits).to(device) 

        xbar_row = math.ceil(pos_bit_slice_weight.shape[0]/cfg.xbar_row_size)
        xbar_col = math.ceil(pos_bit_slice_weight.shape[1]/cfg.xbar_col_size)

        weight_xbar = torch.zeros((2,xbar_row*cfg.xbar_row_size, xbar_col*cfg.xbar_col_size)).to(device)
        weight_xbar[0,:pos_bit_slice_weight.shape[0], :pos_bit_slice_weight.shape[1]] = pos_bit_slice_weight
        weight_xbar[1,:neg_bit_slice_weight.shape[0], :neg_bit_slice_weight.shape[1]] = neg_bit_slice_weight

        bit_slice_num = weight_bits//bit_slice
        bit_stream_num = input_bits//bit_stream
        assert (cfg.xbar_row_size > bit_slice_num), "Attempting zero division, adjust xbar_col_size"
        bias_addr = [weight_channels_out//int(cfg.xbar_col_size/bit_slice_num), weight_channels_out%int(cfg.xbar_col_size/bit_slice_num)]      #####

        #xbars = weight_xbar.unfold(1,cfg.xbar_row_size, cfg.xbar_col_size).unfold(2, cfg.xbar_row_size, cfg.xbar_col_size)
        xbars = weight_xbar.unfold(1,cfg.xbar_row_size, cfg.xbar_row_size).unfold(2, cfg.xbar_col_size, cfg.xbar_col_size)
        
        input_batch = input.shape[0]
        input_channels = input.shape[1]     # weight_channels_in == input_channels
        input_row = input.shape[2] + padding[0]*2
        input_col = input.shape[3] + padding[1]*2
        input_pad = torch.zeros((input_batch, input_channels, input_row, input_col)).to(device)
        input_pad[:,:,padding[0]:input_row-padding[0],padding[1]:input_col-padding[1]] = input
        pos = torch.ones(input_batch*num_pixel, input_channels*weight_row*weight_col).to(device)
        neg = pos.clone().fill_(0)
        
        output_row = (input_row - weight_row)//stride[0] + 1
        output_col = (input_col - weight_col)//stride[1] + 1 
        output = torch.zeros((input_batch, weight_channels_out, output_row, output_col)).to(device)

        flatten_binary_input = torch.zeros(input_batch*num_pixel, xbars.shape[1]*cfg.xbar_row_size, bit_stream_num).to(device)
        flatten_input_sign_temp = torch.zeros(input_batch*num_pixel, xbars.shape[1]*cfg.xbar_row_size, bit_stream_num).to(device)
        flatten_input_sign_xbar= torch.zeros(input_batch*num_pixel, xbars.shape[1],cfg.xbar_row_size, bit_stream_num).to(device)
        
        #variables transferred to GPU
        xbars_row = xbars.shape[1]  # dimension 0 is for sign 
        xbars_col = xbars.shape[2]
        
        zero_mvmtensor = torch.zeros(input_batch*num_pixel, xbars.shape[1],cfg.xbar_row_size, bit_stream_num).to(device)

        shift_add_bit_stream= torch.pow(2*torch.ones(bit_stream_num).float(), bit_stream*torch.arange(0,bit_stream_num).float()).to(device)
        shift_add_bit_slice=  torch.pow(2*torch.ones(bit_slice_num).float(),  bit_slice*torch.arange(bit_slice_num-1, -1, -1).float()).to(device)
        Gon = cfg.Gon
        Goff = cfg.Goff
        Nstates_slice = 2**bit_slice-1        
        if bit_stream ==1:
            shift_add_bit_stream[-1] *= -1        # last bit --> subtract
            shift_add_bit_stream = shift_add_bit_stream.expand((input_batch*num_pixel, xbars_row, xbars_col, cfg.xbar_col_size//bit_slice_num, bit_stream_num)).transpose(3,4).to(device)
            shift_add_bit_slice = shift_add_bit_slice.expand((input_batch*num_pixel, xbars_row, xbars_col, cfg.xbar_col_size//bit_slice_num, bit_slice_num)).to(device)
            output_reg = torch.zeros(input_batch*num_pixel, xbars_row, xbars_col, bit_stream_num, cfg.xbar_col_size//bit_slice_num).to(device) # for 32-fixed  
            G_real0 = (xbars[0]*(Gon - Goff)/Nstates_slice + Goff)
            G_real1 = (xbars[1]*(Gon - Goff)/Nstates_slice + Goff)
        else:
            shift_add_bit_stream = shift_add_bit_stream.expand((2, input_batch*num_pixel, xbars_row, xbars_col, cfg.xbar_col_size//bit_slice_num, bit_stream_num)).transpose(4,5).to(device)
            shift_add_bit_slice = shift_add_bit_slice.expand((2, input_batch*num_pixel, xbars_row, xbars_col, cfg.xbar_col_size//bit_slice_num, bit_slice_num)).to(device)
            output_reg = torch.zeros(2, input_batch*num_pixel, xbars_row, xbars_col, bit_stream_num, cfg.xbar_col_size//bit_slice_num).to(device)
            G_real0 = (xbars[0]*(Gon - Goff)/Nstates_slice +Goff)
            G_real1 = (xbars[1]*(Gon - Goff)/Nstates_slice +Goff)

        #unfold = nn.Unfold(kernel_size=(weight_row, weight_row), stride=(stride[0], stride[1]))
        unfold = nn.Unfold(kernel_size=(weight_row, weight_col), stride=(stride[0], stride[1]))
        
        input_patch_row = (tile_row-1)*stride[0] + weight_row
        stride_input_row = stride[0]*tile_row
        input_patch_col = (tile_col-1)*stride[1] + weight_col
        stride_input_col = stride[1]*tile_col
        
        # Output feature map size should be multiple of tile size
        if (tile_row > output_row):
            tile_row = output_row
        if (tile_col > output_col):
            tile_col = output_col
        assert output_row%tile_row == 0 and output_col%tile_col == 0, "Output feature map size should be multiple of tile size"
        for i in range(math.ceil(output_row/tile_row)):
            for j in range(math.ceil(output_col/tile_col)):
                input_temp = unfold(input_pad[:,:, stride_input_row*i:stride_input_row*i+input_patch_row, stride_input_col*j:stride_input_col*j+input_patch_col]).permute(2,0,1) # #patches, batchsize, k^2*I
                input_temp = input_temp.reshape(input_batch*num_pixel,-1)          #new_batch_size = batch_size*#_of_output_pixel    
                if bit_stream >1:
                    flatten_input_sign = torch.where(input_temp > 0, pos, neg).expand(bit_stream_num,-1,-1).permute(1, 2, 0) 
                    flatten_input_sign_temp[:,:flatten_input_sign.shape[1]] = flatten_input_sign
                    flatten_input_sign_xbar = flatten_input_sign_temp.reshape(input_batch*num_pixel, xbars.shape[1],cfg.xbar_row_size, bit_stream_num)
                    input_temp.abs_()

                flatten_binary_input_temp = float_to_16bits_tensor_fast(input_temp, input_bit_frac, bit_stream, bit_stream_num, input_bits)   # batch x n x 16
                flatten_binary_input[:,:flatten_binary_input_temp.shape[1]] = flatten_binary_input_temp
                flatten_binary_input_xbar = flatten_binary_input.reshape((input_batch*num_pixel, xbars.shape[1],cfg.xbar_row_size, bit_stream_num))  
                

                dataset_ = dataset if (i==0 and j==0)or(i==output_row-1 and j== output_col-1) else False
                xbars_out = mvm_tensor(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, flatten_binary_input_xbar, flatten_input_sign_xbar, 
                                       bias_addr, xbars[0], bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, 
                                       acm_bit_frac, G_real0, dataset_) - \
                            mvm_tensor(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, flatten_binary_input_xbar, flatten_input_sign_xbar,
                                       bias_addr, xbars[1], bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, 
                                       acm_bit_frac, G_real1, False)
                                
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

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'bit_slice', 'bit_stream','weight_bits', 'weight_bit_frac','input_bits', 'input_bit_frac',
                     'adc_bit','acm_bits', 'acm_bit_frac']

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, check_grad=False, bit_slice=2, bit_stream=1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, tile_row=2, tile_col=2, xbmodel=None, xbmodel_weight_path=None, dataset=False):
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
        self.bit_slice = cfg.bit_slice  if cfg.ifglobal_bit_slice else bit_slice
        self.bit_stream = cfg.bit_stream if cfg.ifglobal_bit_stream else bit_stream
        self.weight_bits = cfg.weight_bits if cfg.ifglobal_weight_bits else weight_bits
        self.weight_bit_frac = cfg.weight_bit_frac if cfg.ifglobal_weight_bit_frac else weight_bit_frac
        self.input_bits = cfg.input_bits if cfg.ifglobal_input_bits else input_bits
        self.input_bit_frac = cfg.input_bit_frac if cfg.ifglobal_input_bit_frac else input_bit_frac
        self.adc_bit = cfg.adc_bit if cfg.ifglobal_adc_bit else adc_bit
        self.acm_bits = cfg.acm_bits if cfg.ifglobal_acm_bits else acm_bits
        self.acm_bit_frac = cfg.acm_bit_frac if cfg.ifglobal_acm_bit_frac else acm_bit_frac
        self.xbmodel = cfg.xbmodel if cfg.ifglobal_xbmodel else xbmodel
        self.xbmodel_weight_path = cfg.xbmodel_weight_path if cfg.ifglobal_xbmodel_weight_path else xbmodel_weight_path
        if (cfg.non_ideality):
            assert (self.xbmodel != None)
            assert (self.xbmodel_weight_path != None)
            self.xbmodel.load_state_dict(torch.load(self.xbmodel_weight_path)['state_dict'])
        self.dataset = cfg.dataset if cfg.ifglobal_dataset else xbmodel # flag for dataset collection
        self.tile_col = cfg.tile_col if cfg.ifglobal_tile_col else tile_col
        self.tile_row = cfg.tile_row if cfg.ifglobal_tile_row else tile_row

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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', check_grad=False, bit_slice=2, bit_stream=1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, tile_row=2, tile_col=2, xbmodel=None, xbmodel_weight_path=None, dataset=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(Conv2d_mvm, self).__init__( in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, tile_row, tile_col, xbmodel, xbmodel_weight_path, dataset)
    #@weak_script_method
    def forward(self, input):
            return Conv2d_mvm_function.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.bit_slice, self.bit_stream, self.weight_bits, self.weight_bit_frac, self.input_bits, self.input_bit_frac, self.adc_bit, self.acm_bits, self.acm_bit_frac, self.tile_row, self.tile_col, self.xbmodel, self.xbmodel_weight_path, self.dataset)


class Linear_mvm_function(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, 
                bit_slice=2, bit_stream=1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, xbmodel=None, xbmodel_weight_path=None, dataset=False):

        if weight_bit_frac == -1:
            weight_bit_frac = weight_bits//4*3
        if input_bit_frac == -1:
            input_bit_frac = input_bits//4*3
        if acm_bit_frac == -1:
            acm_bit_frac = acm_bits//4*3      
        if adc_bit == -1:
            adc_bit = int(math.log2(cfg.xbar_row_size))
            if bit_stream != 1:
                adc_bit += bit_stream
            if bit_slice != 1:
                adc_bit += bit_slice
        device = input.device
        weight_channels_out = weight.shape[0]
        weight_channels_in = weight.shape[1]
        pos_weight = torch.clamp(weight, min=0)
        neg_weight = torch.clamp(weight, max=0).abs()

        pos_bit_slice_weight = bit_slicing(pos_weight, weight_bit_frac, bit_slice, weight_bits) ## v2: flatten weights --> fixed point --> bit slice -- v1
        neg_bit_slice_weight = bit_slicing(neg_weight, weight_bit_frac, bit_slice, weight_bits) ## 

        # bitsliced weight into 128x128 xbars 
        # xbar_row separates inputs --> results in a same column with different rows will be added later
        xbar_row = math.ceil(pos_bit_slice_weight.shape[0]/cfg.xbar_row_size)
        xbar_col = math.ceil(pos_bit_slice_weight.shape[1]/cfg.xbar_col_size)

        weight_xbar = torch.zeros((2,xbar_row*cfg.xbar_row_size, xbar_col*cfg.xbar_col_size)).to(device)
        weight_xbar[0,:pos_bit_slice_weight.shape[0], :pos_bit_slice_weight.shape[1]] = pos_bit_slice_weight
        weight_xbar[1,:neg_bit_slice_weight.shape[0], :neg_bit_slice_weight.shape[1]] = neg_bit_slice_weight

        xbars = torch.zeros((2,xbar_row, xbar_col, cfg.xbar_row_size, cfg.xbar_col_size)).to(device)

        bit_slice_num = weight_bits//bit_slice
        bit_stream_num = input_bits//bit_stream

        bias_addr = [weight_channels_out//int(cfg.xbar_col_size/bit_slice_num), weight_channels_out%int(cfg.xbar_col_size/bit_slice_num)]      #####
        for i in range(xbar_row):
            for j in range(xbar_col):
                for k in range(2):
                    xbars[k,i,j] = weight_xbar[k,i*cfg.xbar_row_size:(i+1)*cfg.xbar_row_size, j*cfg.xbar_col_size:(j+1)*cfg.xbar_col_size]

        input_batch = input.shape[0]
        input_channels = input.shape[1]     # weight_channels_in == input_channels
        pos = torch.ones(input.shape).to(device)
        neg = pos.clone().fill_(0)      

        binary_input = torch.zeros(input_batch, xbars.shape[1]*cfg.xbar_row_size, bit_stream_num).to(device)
        input_sign_temp = torch.zeros(input_batch, xbars.shape[1]*cfg.xbar_row_size, bit_stream_num).to(device)
        input_sign_xbar = torch.zeros(input_batch, xbars.shape[1],cfg.xbar_row_size, bit_stream_num).to(device)
        
        if bit_stream > 1:
            input_sign = torch.where(input > 0, pos, neg).expand(bit_stream_num, -1, -1).permute(1,2,0)
            input_sign_temp[:,:input_sign.shape[1]] = input_sign
            input_sign_xbar = input_sign_temp.reshape(input_batch, xbars.shape[1],cfg.xbar_row_size, bit_stream_num)
            input.abs_()

        binary_input[:,:input.shape[1]] = float_to_16bits_tensor_fast(input, input_bit_frac, bit_stream, bit_stream_num, input_bits)   # batch x n x 16

        binary_input = binary_input.reshape((input_batch, xbars.shape[1], cfg.xbar_row_size, bit_stream_num))
        
        #initializations brought out of mvm_tensors, since they are only needed once for the output
        xbars_row = xbars.shape[1]
        xbars_col = xbars.shape[2]    
         
        zero_mvmtensor = torch.zeros(input_batch, xbars.shape[1],cfg.xbar_row_size, bit_stream_num).to(device)
        shift_add_bit_stream = torch.zeros(bit_stream_num) # input bits = 16
        for i in range(bit_stream_num):
            shift_add_bit_stream[i] = 2**(bit_stream*i)
        shift_add_bit_slice = torch.zeros(bit_slice_num) # 16bit / 2bit-slice
        for i in range(bit_slice_num):
            shift_add_bit_slice[-i-1] = 2**(bit_slice*i)        

        Gon = cfg.Gon
        Goff = cfg.Goff
        Nstates_slice = 2**bit_slice-1           
        if bit_stream ==1:
            shift_add_bit_stream[-1] *= -1        # last bit --> subtract
            shift_add_bit_stream = shift_add_bit_stream.expand((input_batch, xbars_row, xbars_col, cfg.xbar_col_size//bit_slice_num, bit_stream_num)).transpose(3,4).to(device)
            shift_add_bit_slice = shift_add_bit_slice.expand((input_batch, xbars_row, xbars_col, cfg.xbar_col_size//bit_slice_num, bit_slice_num)).to(device)
            output_reg = torch.zeros(input_batch, xbars_row, xbars_col, bit_stream_num, cfg.xbar_col_size//bit_slice_num).to(device) # for 32-fixed  
            G_real0 = (xbars[0]*(Gon - Goff)/Nstates_slice + Goff)         
            G_real1 = (xbars[1]*(Gon - Goff)/Nstates_slice + Goff)              
        else:
            shift_add_bit_stream = shift_add_bit_stream.expand((2, input_batch, xbars_row, xbars_col, cfg.xbar_col_size//bit_slice_num, bit_stream_num)).transpose(4,5).to(device)
            shift_add_bit_slice = shift_add_bit_slice.expand((2, input_batch, xbars_row, xbars_col, cfg.xbar_col_size//bit_slice_num, bit_slice_num)).to(device)
            output_reg = torch.zeros(2, input_batch, xbars_row, xbars_col, bit_stream_num, cfg.xbar_col_size//bit_slice_num).to(device)   
            G_real0 = (xbars[0]*(Gon - Goff)/Nstates_slice +Goff)
            G_real1 = (xbars[1]*(Gon - Goff)/Nstates_slice +Goff)
                
            xbars_out = mvm_tensor(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, binary_input, input_sign_xbar, bias_addr, xbars[0],
                                   bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, dataset, G_real0) - \
                        mvm_tensor(zero_mvmtensor, shift_add_bit_stream, shift_add_bit_slice, output_reg, binary_input, input_sign_xbar, bias_addr, xbars[1], 
                                   bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bits, acm_bit_frac, dataset, G_real1)

        output = xbars_out[:, :weight_channels_out]
 
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None 

class Linear_mvm(nn.Module):
    def __init__(self, input_features, output_features, bias=True, bit_slice=2, bit_stream=1, weight_bits=16, weight_bit_frac=-1, input_bits=16, input_bit_frac=-1, adc_bit=-1, acm_bits=16, acm_bit_frac=-1, xbmodel=None, xbmodel_weight_path=None):
        super(Linear_mvm, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
            self.bias.data.uniform_(-0.1, 0.1) 
        else:
            self.register_parameter('bias', None)

        # Functional simulator parameters
        self.bit_slice = cfg.bit_slice  if cfg.ifglobal_bit_slice else bit_slice
        self.bit_stream = cfg.bit_stream if cfg.ifglobal_bit_stream else bit_stream
        self.weight_bits = cfg.weight_bits if cfg.ifglobal_weight_bits else weight_bits
        self.weight_bit_frac = cfg.weight_bit_frac if cfg.ifglobal_weight_bit_frac else weight_bit_frac
        self.input_bits = cfg.input_bits if cfg.ifglobal_input_bits else input_bits
        self.input_bit_frac = cfg.input_bit_frac if cfg.ifglobal_input_bit_frac else input_bit_frac
        self.adc_bit = cfg.adc_bit if cfg.ifglobal_adc_bit else adc_bit
        self.acm_bits = cfg.acm_bits if cfg.ifglobal_acm_bits else acm_bits
        self.acm_bit_frac = cfg.acm_bit_frac if cfg.ifglobal_acm_bit_frac else acm_bit_frac
        self.xbmodel = cfg.xbmodel if cfg.ifglobal_xbmodel else xbmodel
        self.xbmodel_weight_path = cfg.xbmodel_weight_path if cfg.ifglobal_xbmodel_weight_path else xbmodel_weight_path
        if (cfg.non_ideality):
            assert (self.xbmodel != None)
            assert (self.xbmodel_weight_path != None)
            self.xbmodel.load_state_dict(torch.load(cfg.pretrained_model_path)['state_dict'])
        self.dataset = cfg.dataset if cfg.ifglobal_dataset else xbmodel # flag for dataset collection

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return Linear_mvm_function.apply(input, self.weight, self.bias, 
        self.bit_slice, self.bit_stream, self.weight_bits, self.weight_bit_frac, self.input_bits, self.input_bit_frac, self.adc_bit, self.acm_bits, self.acm_bit_frac, self.xbmodel, self.xbmodel_weight_path, self.dataset)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
