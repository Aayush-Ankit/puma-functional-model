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

def mvm_tensor(zeros, shift_add_bit_stream, shift_add_bit_slice, output_reg, flatten_input, flatten_input_sign, bias_addr, 
               xbars, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bit, acm_bit_frac): 
#def mvm_tensor(flatten_input, flatten_input_sign, bias_addr, xbars, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bit, acm_bit_frac, device):   # version 2
 
    # xbars shape:          [xbars_row, xbars_col, XBAR_ROW_SIZE, XBAR_COL_SIZE]
    # flatten_input shape:  [batch_size, xbars_row, XBAR_ROW_SIZE, 16]
    # 2-bit bit-slicing
    xbars_row = xbars.shape[0]
    batch_size = flatten_input.shape[0]
    bit_stream_num = input_bits//bit_stream

    if bit_stream == 1:
        for i in range(bit_stream_num): # 16bit input
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

def mvm_tensor_nonid(zeros, shift_add_bit_stream, shift_add_bit_slice, output_reg, output_analog, Goffmat, G_real_flatten, G_real, model, flatten_input,
                   flatten_input_sign, bias_addr, xbars, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bit, 
                   acm_bit_frac):  #### These should be 'almost' completely changed. 

    # xbars shape:          [xbars_row, xbars_col, XBAR_ROW_SIZE, XBAR_COL_SIZE]
    # flatten_input shape:  [batch_size, xbars_row, XBAR_ROW_SIZE, 16]
    # 2-bit bit-slicing

    Gon = 1/100
    Goff = 1/600
    Nstates_slice = 2**bit_slice-1
    Nstates_stream = 2**bit_stream-1
    Vmax = 0.25
#    Goffmat = Goff*torch.ones(XBAR_ROW_SIZE,XBAR_COL_SIZE).to(device)
    Comp_factor = Nstates_slice*Nstates_stream/((Gon-Goff)*Vmax)
    inmax_V = Vmax
    inmin_V = 0
#    #100k
#    inmax_test =    1.2
#    inmin_test =    0.85

    #100k, 128x128
    inmax_test =    1.2905
    inmin_test =    0.8
        #64x64
#    inmax_test = 1.0959
#    inmin_test = 0.8
        #32x32
#        inmax_test = 1.0156
#        inmin_test = 0.8
        #16x16
#        inmax_test = 0.9935
#        inmin_test = 0.8  

    #300k
    # inmax_test =    1.27
    # inmin_test =    0.88
    #50k
    # inmax_test =    1.65
    # inmin_test =    1.1
    #16x16
    # inmax_test =    1.22
    # inmin_test =    0.81

    #32x32
    # inmax_test =    1.4
    # inmin_test =    0.92
    #ONOFF2
    # inmax_test = 1.5
    # inmin_test = 1
    #ssw_pt5
    # inmax_test = 1.14;
    # inmin_test = 1.0856;
     #ssw
#    inmax_test = 1.14
#    inmin_test = 1.08
     #100_allpt5
    # inmax_test = 1.3;
    # inmin_test = 0.85;
   # ONOFF10
    # inmax_test = 1.25
    # inmin_test = 0.8
    ###############################
#4bin_2bwt
    # inmax_test = 1.18;
    # inmin_test = 0.8;
#4bin_1bwt
    # inmax_test = 1.2;
    # inmin_test = 0.85;
#1bin_1bwt
    # inmax_test = 1
    # inmin_test = 0.85
#1bin_2bwt
    # inmax_test = 1.1
    # inmin_test = 0.8

     #1bin_4bwt
    # inmax_test = 1.1
    # inmin_test = 0.85
         #2bin_4bwt
    #inmax_test = 1.2
    #inmin_test = 0.8

#2bin_2bwt
    # inmax_test = 1.2
    # inmin_test = 0.85
         #2bin_1bwt
    #inmax_test = 1.2
    #inmin_test = 0.85



    in_diff = inmax_test-inmin_test

    xbars_row = xbars.shape[0]
    xbars_col = xbars.shape[1]
    batch_size = flatten_input.shape[0]

    bit_slice_num = weight_bits//bit_slice
    bit_stream_num = input_bits//bit_stream

    input_pos = torch.where(flatten_input_sign == 1, flatten_input, zeros)
    input_neg = flatten_input.sub(input_pos)
    input_split = torch.stack([input_pos, input_neg])
    if bit_stream == 1:
        V_real = flatten_input*Vmax/Nstates_stream
        V_real_scaled = (V_real-inmin_V)/(inmax_V-inmin_V)
    else:
        V_real = input_split*Vmax/Nstates_stream
        V_real_scaled = (V_real-inmin_V)/(inmax_V-inmin_V)
    
    if bit_stream == 1:
        for i in range(bit_stream_num): # 16bit input
            V_real_loop = V_real[:,:,:,-1-i].reshape((batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1))   #V_real.shape batchsize, xbar_rows, xbarsize, num_bitstreams
            V_real_scaled_loop = V_real_scaled[:,:,:,-1-i].reshape((batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1))
            output_bias_all = torch.sum(torch.mul(Goffmat,V_real_loop),3).unsqueeze(3).expand(batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1)
            output_real_out= torch.mul(G_real, V_real_loop)
            output_real_out = torch.sum(output_real_out,3)
            if cfg.loop == True:
                for xrow in range(xbars_row):
                    for xcol in range(xbars_col):
                        output_real = output_real_out[:,xrow,xcol]
                        input_VG = torch.cat((G_real_flatten[:,xrow,xcol], V_real_scaled_loop[:, xrow, 0]),1)
                        
                        output_niratio = model(input_VG)
                        output_niratio_unscale = (output_niratio) * (inmax_test - inmin_test )  + inmin_test
                        output_bias = output_bias_all[:, xrow, 0].view(batch_size,XBAR_ROW_SIZE)
                        output_nonideal = (output_real-output_bias).div(output_niratio_unscale)
                        output_analog_xbar_real = ((output_nonideal)*Comp_factor)
    
                        # --------------------------------------
                        output_analog[:, xrow, xcol] = output_analog_xbar_real
            else:
                V_int = V_real_scaled_loop.expand(batch_size, xbars_row, xbars_col, XBAR_ROW_SIZE,1)
                input_VG = torch.cat((G_real_flatten, V_int), 3) # concat along the input dim of XB so total inputs = r^2+r
                output_real = output_real_out.permute(1,2,0,3).reshape(output_real_out.shape[0]*output_real_out.shape[1]*output_real_out.shape[2], output_real_out.shape[3]) # reshaped to # currents, col currents 
                input_VG_flatten = input_VG.permute(1,2,0,3,4).reshape(batch_size*input_VG.shape[1]*input_VG.shape[2], input_VG.shape[3])
                output_niratio = model(input_VG_flatten)
                output_niratio_unscale = (output_niratio) * in_diff + inmin_test
                output_bias = output_bias_all.expand(batch_size, output_bias_all.shape[1], xbars_col, XBAR_ROW_SIZE, 1).permute(1,2, 0,3,4).reshape(batch_size*xbars_row*xbars_col,XBAR_ROW_SIZE)
                output_analog_xbar = (output_real-output_bias).div(output_niratio_unscale)
                
                output_analog = torch.stack(torch.split(((output_analog_xbar)*Comp_factor),batch_size,dim=0)).reshape(xbars_row, xbars_col, batch_size, XBAR_COL_SIZE).permute(2,0,1,3)
                        
            #####
            output_analog_=output_analog.reshape(shift_add_bit_slice.shape)
            output_reg[:,:,:,i,:] = torch.sum(torch.mul(output_analog_, shift_add_bit_slice), 4)

        output = torch.sum(torch.mul(output_reg, shift_add_bit_stream), 3)

        output.div_(2**(input_bit_frac + weight_bit_frac - acm_bit_frac)).trunc_()
        output.fmod_(2**acm_bit).div_(2**acm_bit_frac)
        output = torch.sum(output, 1).reshape(batch_size, -1).type(torch.float)
    else:
        for i in range(bit_stream_num): # 16bit input
            V_real_loop = V_real[:,:,:,:,-1-i].reshape((2, batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1))
            V_real_scaled_loop = V_real_scaled[:,:,:,:,-1-i].reshape((2, batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1))
            output_real_out= torch.mul(G_real, V_real_loop)
            output_real_out = torch.sum(output_real_out,4)
            output_bias_all = torch.sum(torch.mul(Goffmat,V_real_loop),4).unsqueeze(4).expand(2,batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1)#.to(device)

            for xsign in range(2):
                if cfg.loop == True:
                    for xrow in range(xbars_row):
                        for xcol in range(xbars_col):
                            output_real = output_real_out[xsign,:,xrow,xcol]
                            input_VG = torch.cat((G_real_flatten[:,xrow,xcol], V_real_scaled_loop[xsign, :, xrow, 0]),1)
                            output_niratio = model(input_VG)
                            output_niratio_unscale = (output_niratio) * in_diff  + inmin_test
                            output_bias = output_bias_all[xsign, :, xrow, 0].view(batch_size,XBAR_ROW_SIZE)
                            output_nonideal = (output_real-output_bias).div(output_niratio_unscale)
                            output_analog_xbar_real = ((output_nonideal)*Comp_factor)
                            output_analog[xsign, :, xrow, xcol] = output_analog_xbar_real
                else:
                    V_int = V_real_scaled_loop.expand(2, batch_size, xbars_row, xbars_col, XBAR_ROW_SIZE,1)
                    input_VG = torch.cat((G_real_flatten, V_int[xsign]), 3)
                    output_real = output_real_out[xsign].permute(1,2,0,3).reshape(output_real_out.shape[1]*output_real_out.shape[2]*output_real_out.shape[3], output_real_out.shape[4])
                    input_VG_flatten = input_VG.permute(1,2,0,3,4).reshape(batch_size*input_VG.shape[1]*input_VG.shape[2], input_VG.shape[3])
                    output_niratio = model(input_VG_flatten)
                    output_niratio_unscale = (output_niratio) * in_diff + inmin_test
                    output_bias = output_bias_all[xsign].expand(batch_size, output_bias_all.shape[2], xbars_col, XBAR_ROW_SIZE, 1).permute(1,2, 0,3,4).reshape(batch_size*xbars_row*xbars_col,XBAR_ROW_SIZE)
                    output_analog_xbar = (output_real-output_bias).div(output_niratio_unscale)
                    output_analog[xsign, :] = torch.stack(torch.split(((output_analog_xbar)*Comp_factor),batch_size,dim=0)).reshape(xbars_row, xbars_col, batch_size, XBAR_COL_SIZE).permute(2,0,1,3)
            
            output_analog_ = output_analog.reshape(shift_add_bit_slice.shape)
            output_reg[:,:,:,:,i,:] = torch.sum(torch.mul(output_analog_, shift_add_bit_slice), 5) # -1
        
        output_split = torch.sum(torch.mul(output_reg, shift_add_bit_stream), 4)

        output_split.div_(2**(input_bit_frac + weight_bit_frac - acm_bit_frac)).trunc_()
        output_split.fmod_(2**acm_bit).div_(2**acm_bit_frac)

        # + sum xbar_rows
        output_split = torch.sum(output_split, 2).reshape(2, batch_size, -1)
        output = output_split[0].sub(output_split[1]).type(torch.float)
    
    #del shift_add_bit_stream, shift_add_bit_slice, output_reg
    return output