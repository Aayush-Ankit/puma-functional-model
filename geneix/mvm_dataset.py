import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import os
import argparse
import pdb

import src.config as cfg

XBAR_ROW_SIZE = cfg.xbar_row_size
XBAR_COL_SIZE = cfg.xbar_col_size

def get_tree_index(idx):    # version 2
    n = idx.shape[1]
    idx = idx.mul(2)
    idx = torch.cat((idx, idx.add(1)))
    idx =idx.reshape((1, n*2))  
    return idx

idx2 = torch.tensor([range(2)])     # [0, 1]
idx4 = get_tree_index(idx2)         # [0, 2, 1, 3]
idx8 = get_tree_index(idx4)         # [0, 4, 2, 6, 1, 5, 3, 7]
idx16 = get_tree_index(idx8)        # [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
idx32 = get_tree_index(idx16)
idxs = [idx2, idx4, idx8, idx16, idx32]

def get_index_rearrange(idx, out_ch):   # version 2
    n =idx.shape[1]
    idx_new = idx.expand(out_ch, n)
    cols = torch.tensor([range(out_ch)])
    cols = torch.tensor([range(out_ch)]).mul(n).transpose(1,0).expand(out_ch,n)
    idx_new = idx.add(cols)
    idx_new = idx_new.transpose(1,0).reshape(1, -1)  
    return idx_new


## 16 bit fixed point
##   +---+--------------------+------------------------+
##   | 1 |  15-n   |                n                  |
##   +---+--------------------+------------------------+
##   sign  integer              fraction
##

def slicing(weight, n_bit): # version 2
    weight_int = torch.floor(weight)
    weight_frac = weight.remainder(1).mul(2**n_bit)
    weight = torch.cat([weight_int, weight_frac])
    return weight

# fix bit_slicing to 2 bit --> later make it variable
def bit_slicing(weight, frac_bit, bit_slice, weight_bits):  # version 2
    #assume positive
    # weight.shape[0] is output channels + 1 (bias)
    # weight.shape[1] is flattened weight length

    # weight_bits = 16 or 32
    int_bit = weight_bits-frac_bit-1
    # clipping
    weight = torch.clamp(weight, -2**int_bit, 2**int_bit-1/2**frac_bit)
    out_channel = weight.shape[0]

    n = weight_bits
    while(n > bit_slice):
        n //= 2
        if n == weight_bits//2:
            weight.mul_(2**(weight_bits//2-int_bit-1))  # 0000. 0000 0000 0000 --> 0000 0000. 0000 0000 // first iteration
        else:
            weight.div_(2**n)
        weight = slicing(weight, n)

    weight[:out_channel].add_(2**bit_slice).fmod_(2**bit_slice) #.fmod_(4)# with bias ## .fmod_(4) # for negative numbers. 4 = 2**2. 
    weight[-out_channel:]= torch.floor(weight[-out_channel:])   # last layer

    # already made 2-bit. -> stop.
    # If I use 2^n bit-slice, I d(on't have to slice more to make 1-bits and then combine it again.
    weight_idx = get_index_rearrange(idxs[int(math.log2(weight_bits//bit_slice))-1], out_channel)
    bitslice = weight.clone()
    bitslice[weight_idx[0],:] = weight
    bitslice = bitslice.t()
    return bitslice

def float_to_16bits_tensor_fast(input, frac_bits, bit_slice, bit_slice_num, input_bits): # input is batch x n tensor / output is n x 16 tensor..
    int_bit = input_bits - frac_bits -1 # extra -1 for sign bit
    #clamp data into the available range
    input = torch.clamp(input, -2**int_bit, 2**int_bit-1/2**frac_bits)
    #normalize
    input = input.div_(2**int_bit)
    #left shift all the fracional values so that all 16 bits comes to the left of decimal
    input = input.mul_(2**(input_bits-1))
    #take the integer part of the input, which represents our 16bit number
    input = torch.floor(input)
    #divide by scalar to get the decimal representation back, MSB----->LSB
    input_sliced = torch.stack([torch.floor(torch.div(input, 2**(i*bit_slice))) - \
                                torch.mul(torch.floor(torch.div(input, 2**((i+1)*bit_slice))), 2**bit_slice) for i in range(bit_slice_num-1,-1,-1) ])
    del input
    return input_sliced.permute(1,2,0)

def mvm_tensor(zeros, shift_add_bit_stream, shift_add_bit_slice, output_reg, flatten_input, flatten_input_sign, bias_addr, xbars, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bit, acm_bit_frac, G_real, dataset): 
    xbars_row = xbars.shape[0]
    xbars_col = xbars.shape[1]
    batch_size = flatten_input.shape[0]
    bit_stream_num = input_bits//bit_stream
    Vmax = cfg.Vmax
    Nstates_stream = 2**bit_stream-1
    
    direc = cfg.direc+'/spice_'+str(XBAR_ROW_SIZE)+'_stream'+str(bit_stream)+'slice'+str(bit_slice)+'_all_layers' 
    if not os.path.exists(direc):
        os.mkdir(direc)
    V=[]
    G=[]
    if bit_stream == 1:
        V_real = flatten_input*Vmax/Nstates_stream
        for i in range(bit_stream_num): # 16bit input 
            if dataset:
                xbars_row1 = xbars.shape[0]
                xbars_col1 = xbars.shape[1]  
                xbars_row1 = cfg.rows if xbars_row1 > 2 else xbars_row  
                xbars_col1 = cfg.cols if xbars_col1 > 2 else xbars_col
             
                for xrow in range(xbars_row1):
                    for xcol in range(xbars_col1):                       
                        v_file_name = direc+'/dataset_V_'+str(XBAR_ROW_SIZE)+'stream'+str(bit_stream)+'slice'+str(bit_slice)+'.txt'
                        g_file_name = direc+'/dataset_G_'+str(XBAR_ROW_SIZE)+'stream'+str(bit_stream)+'slice'+str(bit_slice)+'.txt'
                        G_real_flatten2 = G_real[xrow,xcol].t().reshape(XBAR_ROW_SIZE*XBAR_COL_SIZE).expand(batch_size, XBAR_ROW_SIZE*XBAR_COL_SIZE)
                        V_real_flatten2 = V_real[:, xrow,:,i].view(batch_size,XBAR_ROW_SIZE)
                        V.append(V_real_flatten2.cpu().numpy())
                        G.append(G_real_flatten2.cpu().numpy())
                        with open(v_file_name,'a') as f:
                            np.savetxt(f,V_real_flatten2.cpu().numpy(), delimiter=',')
                        with open(g_file_name,'a') as f:
                            np.savetxt(f,G_real_flatten2.cpu().numpy(), delimiter=',')                          
                        print('data V: ',len(V))
                        print('data G: ',len(G))
            input_stream = flatten_input[:,:,:,-1-i].reshape((batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1))
            #####
            output_analog = torch.mul(xbars, input_stream)
            output_analog = torch.sum(output_analog,3)
            output_analog = torch.clamp(output_analog, min=0, max=2**adc_bit-1)
            #####
            output_analog=output_analog.reshape(shift_add_bit_slice.shape)  # for 32-fixed
            output_reg[:,:,:,i,:] = torch.sum(torch.mul(output_analog, shift_add_bit_slice), 4)

        output = torch.sum(torch.mul(output_reg, shift_add_bit_stream), 3)

        #output = output.float() # uncomment for FP16 (ops on 128, 129 fail without float)
        output.div_(2**(input_bit_frac + weight_bit_frac - acm_bit_frac)).trunc_()
        output.fmod_(2**acm_bit).div_(2**acm_bit_frac)

        # + sum xbar_rows
        output = torch.sum(output, 1).reshape(batch_size, -1).type(torch.float)
    else:
        input_pos = torch.where(flatten_input_sign == 1, flatten_input, zeros)
        input_neg = flatten_input.sub(input_pos)
        input_split = torch.stack([input_pos, input_neg])
        
        for i in range(bit_stream_num): # 16bit input
            input_stream = input_split[:,:,:,:,-1-i].reshape((2, batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1)) #input is arranged from MSB---->LSB
            #####
            output_analog = torch.mul(xbars, input_stream)
            output_analog = torch.sum(output_analog,4)      #sum it along the row dim
            ####
            output_analog=output_analog.reshape(shift_add_bit_slice.shape)
            output_reg[:,:,:,:,i,:] = torch.sum(torch.mul(output_analog, shift_add_bit_slice), 5) # -1 # adding across bit sliced dimension

        output_split = torch.sum(torch.mul(output_reg, shift_add_bit_stream), 4)

        #output_split = output_split.float() # uncomment for FP16 (ops on 149, 150 fail without float)
        output_split.div_(2**(input_bit_frac + weight_bit_frac - acm_bit_frac)).trunc_()
        output_split.fmod_(2**acm_bit).div_(2**acm_bit_frac)

        # + sum xbar_rows
        output_split = torch.sum(output_split, 2).reshape(2, batch_size, -1)
        output = output_split[0].sub(output_split[1])

    #del shift_add_bit_stream, shift_add_bit_slice, output_reg
    return output