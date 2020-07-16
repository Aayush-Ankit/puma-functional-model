import numpy as np
import torch
import math
import pdb
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import os
import argparse
import matplotlib.pyplot as plt 



 
XBAR_COL_SIZE = 64
XBAR_ROW_SIZE = 64
# pretrained_model = torch.load('final_64x64_mlp2layer_xbar_64x64_100_all_dataset_5k_standard_sgd.pth.tar')
# #pretrained_model = torch.load('final_64x64_mlp2layer_xbar_64x64_100_all_low_nonideality_standard_sgd.pth.tar')

# class NN_model(nn.Module):
#     def __init__(self):
#          super(NN_model, self).__init__()
#          self.fc1 = nn.Linear(4160, 10000)
#          self.bn1 = nn.BatchNorm1d(10000)
#          self.relu1 = nn.ReLU(inplace=True)
#          self.do2 = nn.Dropout(0.5)
#          self.fc3 = nn.Linear(10000,64)
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         out = self.fc1(x)
#         out = self.relu1(out)
#         out = self.do2(out)
#         out = self.fc3(out)
#         return out
# #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = NN_model()
# model.cuda() 
# model.eval()
# model.load_state_dict(pretrained_model['state_dict'])

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
def bit_slicing(weight, frac_bit, bit_slice, weight_bits, device):  # version 2

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

    weight[:out_channel].add_(2**(bit_slice-1)) #.fmod_(4)# with bias ## .fmod_(4) # for negative numbers. 4 = 2**2. 
    weight[-out_channel:]= torch.floor(weight[-out_channel:])   # last layer

    # already made 2-bit. -> stop.
    # If I use 2^n bit-slice, I d(on't have to slice more to make 1-bits and then combine it again. 

    weight_idx = get_index_rearrange(idxs[int(math.log2(weight_bits//bit_slice))-1], out_channel)
    bitslice = weight.clone()
    bitslice[weight_idx[0],:] = weight
    bitslice = bitslice.t()

#    del max_weight, min_weight
    torch.cuda.empty_cache()

    return bitslice

def float_to_16bits_tensor(input, frac_bit, bit_stream, input_bits, device): # input is batch x n tensor / output is n x 16 tensor.. version 2

    #assume positive
    # input.shape[0] = batch size
    # input.shape[1] = flattened input length

    int_bit = input_bits-frac_bit-1
    # clipping
#    max_input = torch.ones(input.shape).mul_(2**int_bit-1/2**frac_bit).to(device)
#    min_input = torch.ones(input.shape).mul_(-2**int_bit).to(device)
#    input = torch.where(input < (2**int_bit-1/2**frac_bit), input, max_input)
#    input = torch.where(input > -2**int_bit, input, min_input)
    input=torch.clamp(input, -2**int_bit, 2**int_bit-1/2**frac_bit)
    batch_size = input.shape[0] 

    n = input_bits
    while(n > bit_stream):
        n //= 2
        if n == input_bits//2:
            input.mul_(2**(input_bits//2-int_bit-1))  # 0000. 0000 0000 0000 --> 0000 0000. 0000 0000 // first iteration
        else:
            input.div_(2**n)
        input = slicing(input, n)


    input[:batch_size].add_(2**bit_stream).fmod_(2**bit_stream) # for negative numbers.  
    input[-batch_size:]= torch.floor(input[-batch_size:])   # last layer

    input_idx = get_index_rearrange(idxs[int(math.log2(input_bits//bit_stream)-1)], batch_size)
    bit_slice = input.clone()
    bit_slice[input_idx[0]] = input
    bit_slice = bit_slice.reshape(batch_size, input_bits//bit_stream, -1).transpose(1,2)

#    del max_input, min_input
    torch.cuda.empty_cache()

    return bit_slice


def mvm_tensor(flatten_input, flatten_input_sign, bias_addr, xbars, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bit, acm_bit_frac, device):   # version 2

    # xbars shape:          [xbars_row, xbars_col, XBAR_ROW_SIZE, XBAR_COL_SIZE]
    # flatten_input shape:  [batch_size, xbars_row, XBAR_ROW_SIZE, 16]
    # 2-bit bit-slicing
    xbars_row = xbars.shape[0]
    xbars_col = xbars.shape[1]
    batch_size = flatten_input.shape[0]

    bit_slice_num = weight_bits//bit_slice
    bit_stream_num = input_bits//bit_stream

    zeros = torch.zeros(flatten_input.shape).to(device)
    input_pos = torch.where(flatten_input_sign == 1, flatten_input, zeros)
    input_neg = flatten_input.sub(input_pos)
    input_split = torch.stack([input_pos, input_neg])

    shift_add_bit_stream = torch.zeros(bit_stream_num, dtype=torch.double) # input bits = 16
    for i in range(bit_stream_num):
        shift_add_bit_stream[i] = 2**(bit_stream*i)

    shift_add_bit_slice = torch.zeros(bit_slice_num, dtype=torch.double) # 16bit / 2bit-slice
    for i in range(bit_slice_num):
        shift_add_bit_slice[-i-1] = 2**(bit_slice*i)

    if bit_stream == 1:
        shift_add_bit_stream[-1] *= -1        # last bit --> subtract
        shift_add_bit_stream = shift_add_bit_stream.expand((batch_size, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_stream_num)).transpose(3,4).to(device)
        shift_add_bit_slice = shift_add_bit_slice.expand((batch_size, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_slice_num)).to(device)
        output_reg = torch.zeros(batch_size, xbars_row, xbars_col, bit_stream_num, XBAR_COL_SIZE//bit_slice_num, dtype=torch.double).to(device) # for 32-fixed

        for i in range(bit_stream_num): # 16bit input
            input_stream = flatten_input[:,:,:,-1-i].reshape((batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1))
            #####
            output_analog = torch.mul(xbars, input_stream)
            output_analog = torch.sum(output_analog,3)
            output_analog = torch.clamp(output_analog, min=0, max=2**adc_bit-1)
            #####
            output_analog=output_analog.reshape(shift_add_bit_slice.shape).type(torch.double)   # for 32-fixed
            output_reg[:,:,:,i,:] = torch.sum(torch.mul(output_analog, shift_add_bit_slice), 4)

        output = torch.sum(torch.mul(output_reg, shift_add_bit_stream), 3)
        subt = output[:, :, bias_addr[0], bias_addr[1]].expand(output.shape[2], output.shape[3],-1,-1).permute(2,3,0,1)
        output = output.sub(subt)

        output.div_(2**(input_bit_frac + weight_bit_frac - acm_bit_frac)).trunc_()
        output.fmod_(2**acm_bit).div_(2**acm_bit_frac)


        # + sum xbar_rows
        output = torch.sum(output, 1).reshape(batch_size, -1).type(torch.float)
    else:
        shift_add_bit_stream = shift_add_bit_stream.expand((2, batch_size, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_stream_num)).transpose(4,5).to(device)
        shift_add_bit_slice = shift_add_bit_slice.expand((2, batch_size, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_slice_num)).to(device)
        output_reg = torch.zeros(2, batch_size, xbars_row, xbars_col, bit_stream_num, XBAR_COL_SIZE//bit_slice_num, dtype=torch.double).to(device)

        for i in range(bit_stream_num): # 16bit input
            input_stream = input_split[:,:,:,:,-1-i].reshape((2, batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1))
            #####
            output_analog = torch.mul(xbars, input_stream)
            output_analog = torch.sum(output_analog,4)
            output_analog = torch.clamp(output_analog, min=0, max=2**adc_bit-1)
            ####
            output_analog=output_analog.reshape(shift_add_bit_slice.shape).type(torch.double) 
            output_reg[:,:,:,:,i,:] = torch.sum(torch.mul(output_analog, shift_add_bit_slice), 5) # -1

        output_split = torch.sum(torch.mul(output_reg, shift_add_bit_stream), 4)
        subt = output_split[:, :, :, bias_addr[0], bias_addr[1]].expand(output_split.shape[3], output_split.shape[4],-1,-1, -1).permute(2,3,4,0,1)
        output_split = output_split.sub(subt)

        output_split.div_(2**(input_bit_frac + weight_bit_frac - acm_bit_frac)).trunc_()
        output_split.fmod_(2**acm_bit).div_(2**acm_bit_frac)

        # + sum xbar_rows
        output_split = torch.sum(output_split, 2).reshape(2, batch_size, -1)
        output = output_split[0].sub(output_split[1]).type(torch.float)


    del shift_add_bit_stream, shift_add_bit_slice, output_reg
    torch.cuda.empty_cache()

    return output


def mvm_tensor_ind(model, flatten_input, flatten_input_sign, bias_addr, xbars, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bit, acm_bit_frac, device):  #### These should be 'almost' completely changed. 

    # xbars shape:          [xbars_row, xbars_col, XBAR_ROW_SIZE, XBAR_COL_SIZE]
    # flatten_input shape:  [batch_size, xbars_row, XBAR_ROW_SIZE, 16]
    # 2-bit bit-slicing
    Gon = 1/100
    Goff = 1/600
    Nstates_slice = 15
    Nstates_stream = 15
    Vmax = 0.25
    Goffmat = Goff*torch.ones(XBAR_ROW_SIZE,XBAR_COL_SIZE).to(device)
    Comp_factor = (Nstates_slice*Nstates_stream)/((Gon-Goff)*Vmax) 
    inmax_V = Vmax
    inmin_V = 0
    #100_all
    inmax_test =    1.6985
    inmin_test =    1.1221


    xbars_row = xbars.shape[0]
    xbars_col = xbars.shape[1]
    batch_size = flatten_input.shape[0]

    bit_slice_num = weight_bits//bit_slice
    bit_stream_num = input_bits//bit_stream

    zeros = torch.zeros(flatten_input.shape).to(device)
    input_pos = torch.where(flatten_input_sign == 1, flatten_input, zeros)
    input_neg = flatten_input.sub(input_pos)
    input_split = torch.stack([input_pos, input_neg])

    shift_add_bit_stream = torch.zeros(bit_stream_num, dtype=torch.double) # input bits = 16
    for i in range(bit_stream_num):
        shift_add_bit_stream[i] = 2**(bit_stream*i)

    shift_add_bit_slice = torch.zeros(bit_slice_num, dtype=torch.double) # 16bit / 2bit-slice
    for i in range(bit_slice_num):
        shift_add_bit_slice[-i-1] = 2**(bit_slice*i)
    

    if bit_stream == 1:
        shift_add_bit_stream[-1] *= -1        # last bit --> subtract
        shift_add_bit_stream = shift_add_bit_stream.expand((batch_size, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_stream_num)).transpose(3,4).to(device)
        shift_add_bit_slice = shift_add_bit_slice.expand((batch_size, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_slice_num)).to(device)
        output_reg = torch.zeros(batch_size, xbars_row, xbars_col, bit_stream_num, XBAR_COL_SIZE//bit_slice_num, dtype=torch.double).to(device) # for 32-fixed
        output_analog = torch.zeros(batch_size, xbars_row, xbars_col, XBAR_COL_SIZE, dtype=torch.double).to(device)
        #output_analog1 = torch.zeros(batch_size, xbars_row, xbars_col, XBAR_COL_SIZE).to(device)
        for i in range(bit_stream_num): # 16bit input
            input_stream = flatten_input[:,:,:,-1-i].reshape((batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1))
            #####
            Goffmat = Goff*torch.ones(batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1).to(device)
            V_real = input_stream*Vmax/Nstates_stream
            V_real_scaled = [(V_real-inmin_V)/(inmax_V-inmin_V)]
            G_real = xbars*(Gon - Goff)/Nstates_slice +Goff
            G_real_scaled = [(G_real-Goff)/(Gon-Goff)]
            output_bias_all = torch.sum(torch.mul(Goffmat,V_real),3).unsqueeze(3).expand(batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1)
            for xrow in range(xbars_row):
                for xcol in range(xbars_col):
                    # ----- Put your own function here -----
                    #input_t = input_1bit[:,xrow,0].t()
                    #output_analog_xbar = input_t.mm(xbars[xrow, xcol]) #edit IC
                    
                    #----------------V, G Conversion Start--------------------
                    #t1 = time.time()
                    output_real = torch.mul(G_real[xrow,xcol], V_real[xsign, :, xrow, 0])
                    output_real = torch.sum(output_real,1)
                    #G_real_flatten = G_real[xrow,xcol].t().reshape(XBAR_ROW_SIZE*XBAR_COL_SIZE)
                    #G_real_flatten = G_real_flatten.unsqueeze(2).expand(batch_size, XBAR_ROW_SIZE*XBAR_COL_SIZE, 1)
                    #input_VG = torch.cat(V_real[:, xrow, 0], G_real_flatten)
                    #output_niratio = model(input_VG)
                    #output_bias = torch.mul(Goffmat,V_real[:,xrow,0])
                    output_bias = output_bias_all[xsign, :, xrow, 0].view(batch_size,XBAR_ROW_SIZE)
                    output_analog_xbar_real = ((output_real-output_bias)*Comp_factor)
                    #output_analog_xbar_real = ((output_real-output_bias)*Comp_factor)
                    #output_analog_xbar_real = torch.round(output_analog_xbar_real)
                    #t2 = time.time()
                    #print('Time taken for V-G', t2-t1)

                    #----------------V, G Conversion End--------------------

                    #t1 = time.time()

                    #output_analog_xbar = torch.mul(xbars[xrow, xcol], input_stream[:, xrow, 0])   # Product of each elements : [128 x 128]
                    #output_analog_xbar = torch.sum(output_analog_xbar, 1)    # output of one xbar : array of 128 currents
                    #t2 = time.time()
                    #print('Time taken for normal', t2-t1)
                    #input()
                    # --------------------------------------
                    output_analog[:, xrow, xcol] = output_analog_xbar
            #output_analog = torch.mul(xbars, input_stream)
            #output_analog = torch.sum(output_analog,3)
            #####
            output_analog_ = torch.round(output_analog)
            output_analog=output_analog.reshape(shift_add_bit_slice.shape).type(torch.double) 
            output_reg[:,:,:,i,:] = torch.sum(torch.mul(output_analog, shift_add_bit_slice), 4)

        output = torch.sum(torch.mul(output_reg, shift_add_bit_stream), 3)
        subt = output[:, :, bias_addr[0], bias_addr[1]].expand(output.shape[2], output.shape[3],-1,-1).permute(2,3,0,1)
        output = output.sub(subt)

        output.div_(2**(input_bit_frac + weight_bit_frac - acm_bit_frac)).trunc_()
        output.fmod_(2**acm_bit).div_(2**acm_bit_frac)


        # + sum xbar_rows
        output = torch.sum(output, 1).reshape(batch_size, -1).type(torch.float)
    else:
        shift_add_bit_stream = shift_add_bit_stream.expand((2, batch_size, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_stream_num)).transpose(4,5).to(device)
        shift_add_bit_slice = shift_add_bit_slice.expand((2, batch_size, xbars_row, xbars_col, XBAR_COL_SIZE//bit_slice_num, bit_slice_num)).to(device)
        output_reg = torch.zeros(2, batch_size, xbars_row, xbars_col, bit_stream_num, XBAR_COL_SIZE//bit_slice_num, dtype=torch.double).to(device)
        output_reg2 = torch.zeros(2, batch_size, xbars_row, xbars_col, bit_stream_num, XBAR_COL_SIZE//bit_slice_num, dtype=torch.double).to(device)
        output_analog = torch.zeros(2, batch_size, xbars_row, xbars_col, XBAR_COL_SIZE, dtype=torch.double).to(device)
        output_analog2 = torch.zeros(2, batch_size, xbars_row, xbars_col, XBAR_COL_SIZE, dtype=torch.double).to(device)
        for i in range(bit_stream_num): # 16bit input
            input_stream = input_split[:,:,:,:,-1-i].reshape((2, batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1))
            #####
            Goffmat = Goff*torch.ones(2,batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1).to(device)
            V_real = input_stream*Vmax/Nstates_stream
            V_real_scaled = (V_real-inmin_V)/(inmax_V-inmin_V)
            G_real = xbars*(Gon - Goff)/Nstates_slice +Goff
            G_real_scaled = (G_real-Goff)/(Gon-Goff)
            G_real_flatten = G_real_scaled.permute(0,1,3,2).reshape(xbars_row,xbars_col,XBAR_ROW_SIZE*XBAR_COL_SIZE)
            G_real_flatten = G_real_flatten.unsqueeze(3).expand(batch_size, xbars_row,xbars_col, XBAR_ROW_SIZE*XBAR_COL_SIZE, 1)
            output_real_out= torch.mul(G_real, V_real)
            output_real_out = torch.sum(output_real_out,4)
            output_bias_all = torch.sum(torch.mul(Goffmat,V_real),4).unsqueeze(4).expand(2,batch_size, xbars_row, 1, XBAR_ROW_SIZE, 1)
            input_VG = torch.zeros(batch_size, xbars_row, xbars_col, XBAR_ROW_SIZE*XBAR_COL_SIZE+XBAR_ROW_SIZE,1)
            for xsign in range(2):
                for xrow in range(xbars_row):
                    for xcol in range(xbars_col):
                        # ----- Put your own function here -----
                        #input_t = input_1bit[:,xrow,0].t()
                        #output_analog_xbar = input_t.mm(xbars[xrow, xcol]) #edit IC
                        
                        #----------------V, G Conversion Start--------------------
                        #t1 = time.time()
                        #output_real = torch.mul(G_real[xrow,xcol], V_real[xsign, :, xrow, 0])
                        #output_real = torch.sum(output_real,1)
                        output_real = output_real_out[xsign,:,xrow,xcol]
                        # G_real_flatten2 = G_real[xrow,xcol].t().reshape(XBAR_ROW_SIZE*XBAR_COL_SIZE)
                        # V_real_flatten2 = V_real[xsign, :, xrow, 0].view(batch_size, XBAR_ROW_SIZE)
                        # with open('dataset_V_out.txt','a') as f:
                        #     np.savetxt(f,V_real_flatten2.cpu().numpy(), delimiter=',')
                        # with open('dataset_G_out.txt','a') as f:
                        #     np.savetxt(f,G_real_flatten2.cpu().numpy(), delimiter=',')
                        #t = time.time()
                        #G_real_flatten = G_real_scaled[xrow,xcol].t().reshape(XBAR_ROW_SIZE*XBAR_COL_SIZE)
                        # t1 = time.time()
                        # print('Flatten time: ', t1-t)
                        #G_real_flatten = G_real_flatten.unsqueeze(1).expand(batch_size, XBAR_ROW_SIZE*XBAR_COL_SIZE, 1)
                        # t2 = time.time()
                        # print('Expand time: ', t2-t1)
                        #input_VG = []
                        #input_VG[:,xrow,xcol,:] = torch.cat((G_real_flatten[:,xrow,xcol], V_real_scaled[xsign, :, xrow, 0]),1)

                        
                        # # t3 = time.time()
                        # # print('Cat time: ', t3-t2)
                        # output_niratio = model(input_VG)
                        # # t4 = time.time()
                        # # print('model time: ', t4-t3)
                        # output_niratio_unscale = (output_niratio) * (inmax_test - inmin_test )  + inmin_test
                        # # t5 = time.time()
                        # # print('Unscale time: ', t5-t4)
                        
                        # # t6 = time.time()
                        # # print('Vector divide time: ', t6-t5)
                        # #output_niratio_unscale = 1.05*torch.ones(output_niratio_unscale.shape[0], output_niratio_unscale.shape[1])
                        # #print(torch.mean(abs(output_niratio_unscale)))
                        # #output_bias = torch.mul(Goffmat,V_real[:,xrow,0])
                        output_bias = output_bias_all[xsign, :, xrow, 0].view(batch_size,XBAR_ROW_SIZE)
                        # #output_nonideal = (output_real).div((output_niratio_unscale-0.36))-output_bias/torch.mean(output_niratio_unscale-0.36)
                        # #output_niratio_unscale2 = torch.FloatTensor(output_niratio_unscale.shape[0], output_niratio_unscale.shape[1]).normal_(1.4, 0.025).to(device)
                        # output_nonideal = (output_real-output_bias).div((output_niratio_unscale-0.36))
                        # # plt.figure(1)
                        # # plt.hist(output_niratio_unscale2.cpu().numpy())
                       
                        # # plt.figure(2)
                        # # plt.hist(output_niratio_unscale.cpu().numpy())
                        # # plt.show()
                        output_analog_xbar_real2 = ((output_real-output_bias)*Comp_factor)
                        # output_analog_xbar_real = ((output_nonideal)*Comp_factor)
                        # # print('bit_stream_num = ', i, 'xsign = ', xsign, 'xrow = ', xrow, 'xcol = ', xcol, torch.mean(abs(output_analog_xbar_real-output_analog_xbar_real2)))
                        # #output_analog_xbar_real2 = ((output_nonideal2)*Comp_factor)

                        # #output_analog_xbar_real = torch.round(output_analog_xbar_real)
                        # #t2 = time.time()
                        # #print('Time taken for V-G', t2-t1)

                        # #----------------V, G Conversion End--------------------

                        # #t1 = time.time()

                        # #output_analog_xbar = torch.mul(xbars[xrow, xcol], input_stream[xsign, :, xrow, 0])   # Product of each elements : [128 x 128]
                        # #output_analog_xbar = torch.sum(output_analog_xbar, 1)    # output of one xbar : array of 128 currents
                        # #t2 = time.time()
                        # #print('Time taken for normal', t2-t1)
                        # #input()
                        # # --------------------------------------
                        # output_analog[xsign, :, xrow, xcol] = output_analog_xbar_real
                        output_analog[xsign, :, xrow, xcol] = output_analog_xbar_real2

                # output_real = output_real_out[xsign].permute(1,2,0,3).reshape(output_real_out.shape[1]*output_real_out.shape[2]*output_real_out.shape[3], output_real_out.shape[4])
                # #input_VG_flatten = input_VG.permute(1,2,0,3,4).reshape(batch_size*input_VG.shape[1]*input_VG.shape[2], input_VG.shape[3])
                # #output_niratio = model(input_VG)
                # #output_niratio_unscale = (output_niratio) * (inmax_test - inmin_test )  + inmin_test
                # output_bias = output_bias_all[xsign].view(batch_size*output_bias_all.shape[2],XBAR_ROW_SIZE).unsqueeze(1).expand(batch_size*output_bias_all.shape[2], xbars_col, XBAR_ROW_SIZE).permute(1,0,2).reshape(batch_size*xbars_row*xbars_col,XBAR_ROW_SIZE)
                # #output_nonideal = (output_real-output_bias).div((output_niratio_unscale-0.36))
                # output_analog[xsign, :] = torch.stack(torch.split(((output_real-output_bias)*Comp_factor),batch_size,dim=0)).reshape(xbars_row, xbars_col, batch_size, XBAR_COL_SIZE).permute(2,0,1,3)

                



            #output_analog = torch.mul(xbars, input_stream)
            #output_analog = torch.sum(output_analog,3)
            #output_analog2 = torch.mul(xbars, input_stream)
            #output_analog2 = torch.sum(output_analog2,4)
            # print(torch.mean(abs(output_analog-output_analog2)))
            ####
            #output_analog = output_analog[0,:,:,:] - output_analog[1,:,:,:]
            #output_analog = torch.clamp(output_analog, min=0, max=2**adc_bit-1)
            # input_VG2 = input_VG.view(batch_size*xbars_row*xbars_col,XBAR_ROW_SIZE*XBAR_COL_SIZE+XBAR_ROW_SIZE)
            # output_analog2 = torch.mul(xbars, input_stream)
            # output_analog2 = torch.sum(output_analog2,4)
            output_analog_ = output_analog.reshape(shift_add_bit_slice.shape).type(torch.double) 
            output_reg[:,:,:,:,i,:] = torch.sum(torch.mul(output_analog_, shift_add_bit_slice), 5) # -1
            #output_analog2_ = output_analog2.reshape(shift_add_bit_slice.shape).type(torch.double) 
            #output_reg2[:,:,:,:,i,:] = torch.sum(torch.mul(output_analog2_, shift_add_bit_slice), 5) # -1

        output_split = torch.sum(torch.mul(output_reg, shift_add_bit_stream), 4)
        #output_split2 = torch.sum(torch.mul(output_reg2, shift_add_bit_stream), 4)
        # output_split[abs(output_split)<1e6]=0
        subt = output_split[:, :, :, bias_addr[0], bias_addr[1]].expand(output_split.shape[3], output_split.shape[4],-1,-1, -1).permute(2,3,4,0,1)
        #subt2 = output_split2[:, :, :, bias_addr[0], bias_addr[1]].expand(output_split2.shape[3], output_split2.shape[4],-1,-1, -1).permute(2,3,4,0,1)
        output_split = output_split.sub(subt)
        #output_split2 = output_split2.sub(subt2)
        # output_split[abs(output_split)<1e6]=0

        output_split.div_(2**(input_bit_frac + weight_bit_frac - acm_bit_frac)).trunc_()
        output_split.fmod_(2**acm_bit).div_(2**acm_bit_frac)

        #output_split2.div_(2**(input_bit_frac + weight_bit_frac - acm_bit_frac)).trunc_()
        #output_split2.fmod_(2**acm_bit).div_(2**acm_bit_frac)

        # + sum xbar_rows
        
        output_split = torch.sum(output_split, 2).reshape(2, batch_size, -1)
        # output_split2 = torch.sum(output_split2, 2).reshape(2, batch_size, -1)


        output = output_split[0].sub(output_split[1]).type(torch.float)
        # output2 = output_split2[0].sub(output_split2[1]).type(torch.float)


# ---------------------------------------------------------- For Indranil & Mustafa -------------------------------------------------------------
    del shift_add_bit_stream, shift_add_bit_slice, output_reg
    torch.cuda.empty_cache()
    return output

    # return output, output2

##############################################################################################################################
# Below is version 1
#
#
# float --> 16bit fixed point. ex) [1, 0, 1, 0, 0, ..., 0, 1]
def float_to_16bits(number):
    if number >= 8 or number < -8:
        raise ValueError("fixed-point 16bit number should be in -8 <= x < 8")

    bin16 = torch.zeros(16)
    negative = False
    if number < 0:
        negative = True
    number = abs(number)
 
    i = 15
    number *= 2<<11 
    while number > 0:
        if number%2 > 0:
            bin16[i] = 1
        number //= 2
        i -= 1

    if negative:
        flag = True
        i = 15
        while i >= 0:
            if flag == True and bin16[i] == 1:
                flag = False
            elif flag == False:
                bin16[i] = 1 - bin16[i]
            i -= 1
                
    return bin16



def bits16_to_float(bits):
    
    if bits.shape[0] != 16:
        raise ValueError("It should be 16 bits")

    negative = False
    if bits[0] == 1:
        negative = True
        flag = True
        i=15
        while i>=0:
            if flag == True and bits[i] == 1:
                flag = False
            elif flag == False:
                bits[i] = 1 - bits[i]
            i -= 1
    
    number = 0
    for i in range(15):
        number += bits[i+1]
        number *= 2
    
    number /= 2**13
    
    if negative == True:
        number = -number
    
    return number

def bits_to_uint(bits):
    
    length = bits.shape[0]
    number = 0
    for i in range(length-1):
        number += bits[i]
        number *= 2
    number += bits[length-1]
        
    return number  

# Function of ADC
def uint_to_bits(number, nbits=9): # adc produce 9-bits
    
    bits = torch.zeros(nbits)
    for i in range(nbits):
        bits[nbits-1-i] = number%2
        number //= 2
#    print(bits)
    return bits


def binary_add(a,b):  # a and b should be array of 0,1
                      # ex) a=[0], b=[1], c=[0,1]
    len_a = a.shape[0]
    len_b = b.shape[0]
    length = max(len_a,len_b)
    output = torch.zeros(length)

    c = 0
    for i in range(min(len_a,len_b)):
        out = a[len_a-i-1]+b[len_b-i-1]+c
        if out ==3:
            output[length-i-1] = 1
            c = 1
        elif out == 2:
            output[length-i-1] = 0
            c = 1
        elif out == 1:
            output[length-i-1] = 1
            c = 0
        else:
            output[length-i-1] = 0
            c = 0
    #output[0] += c
#    print(a)
#    print(b)
#    print(output)
#    print("--------")
    return output
  
def binary_subtract(a,b):  #a-b
#    print("subtract!!") 
    length = a.shape[0]
    if length != b.shape[0]:
        raise ValueError("Error: Length should be equal")
    output = torch.zeros(length)
    i = length-1
    
    c = 0
    while i>=0:
        if a[i]-b[i]-c == 1:
            output[i] = 1
            c = 0
        elif a[i]-b[i]-c == 0:
            output[i] = 0
            c = 0
        elif a[i]-b[i]-c == -1:
            output[i] = 1
            c = 1
        else:
            output[i] = 0
            c = 1
        i-=1  
    return output  

# Return Bit sliced weights as integer arrays 
# Because the current of each bitline of crossbar is analog value
def bit_slice_weight(weights, nbit): # weights --> 16-bit fixed-point --> nbit (nbit%2 => 0)
    weights = torch.transpose((weights),1,0)
    rows = weights.shape[0]
    cols = weights.shape[1]
    cells_per_weight = int(16/nbit)
  
    bit_sliced_weights = torch.zeros((rows, cols*cells_per_weight))
    for i in range(rows):
        for j in range(cols):
            fix16b_weight = float_to_16bits(weights[i,j])
            for n in range(cells_per_weight):
                bit_sliced_weights[i, j*cells_per_weight +n] = bits_to_uint(fix16b_weight[2*n:2*n+2])
  
    return bit_sliced_weights


# Return output for 1 bit input.
# Values after ADC are shifted and added --> value for Output Register
def adc_shift_n_add(adc_out, nbits=2): # shift and add 8 x 9bit after adc (unsigned)
    n_cells = adc_out.shape[0]
    adc_bits = adc_out.shape[1]
    
    output_bit_len = int(adc_bits + nbits*(n_cells-1)) #23
    output = torch.zeros(output_bit_len)
    output1 = torch.zeros(output_bit_len)
    output2 = torch.zeros(output_bit_len)

    output[:adc_bits] = adc_out[n_cells-1]
    for i in range(1,n_cells):
        
        # shift right nbits
        output1 = output1.fill_(0)
        output1[nbits:output_bit_len] = output[:output_bit_len-nbits]
#        
        # add
        output2[:adc_bits] = adc_out[n_cells-1-i]
        output = binary_add(output1, output2)
    return output


# 1 input set: fixed-point array [n x 16], 
# 1 weight set: Bitsliced float array [n x (16/nbit)] 
# --> output: 1 float number
def mvm_simple(input_bits, bit_sliced_weights):  
    adc_bit = 9
    n_cell = bit_sliced_weights.shape[1]  # 8
    nbits = int(16/n_cell) # 2bit 
    output_analog = torch.zeros(n_cell)
    output_digital = torch.zeros((n_cell, adc_bit))
    output_register = torch.zeros(38)
    output_shift = torch.zeros(38)
    temp = torch.zeros(38)
    for i in range(input_bits.shape[1]):    #16):
        for w_i in range(n_cell):      #8):
            output_analog[w_i] = torch.sum(input_bits[:,15-i]*bit_sliced_weights[:,w_i])
            
            # ADC
            output_digital[w_i] = uint_to_bits(output_analog[w_i])
        temp[:23] = adc_shift_n_add(output_digital, nbits)            
        # shift
        output_shift = output_shift.fill_(0)#torch.clone(output_register)
        output_shift[1:38] = output_register[0:37]
        
        
        # add
        if i < input_bits.shape[1]-1:
            output_register = binary_add(output_shift, temp)
        #else: # subtract (the last input bit)    
        elif i == input_bits.shape[1]-1:
            output_register = binary_subtract(output_shift, temp)
#    print(output_register)
    out = bits16_to_float(output_register[10:26])

    return out





class xbar:  

    def __init__(self, weight): # get bitsliced weight and store it. 

        if weight.shape[0]>128 or weight.shape[1]>128:
            raise ValueError("One Crossbar shape should be < (128,128)")
        
        self.weight = torch.zeros((128,128))
        self.weight[:weight.shape[0], :weight.shape[1]] = weight

    def mvm(self, input): # get input (16-bit) and mvm 

        if input.shape[0]>128:
            raise ValueError("input size should be < 128")
        
        output=torch.zeros(16) # 128/8cells = 16 weights
        for i in range(16):
            output[i] = mvm_simple(input, self.weight[:,i*8:i*8+8])
        
        return output





