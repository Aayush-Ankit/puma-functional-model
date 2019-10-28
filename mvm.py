import numpy as np
import torch
import torch.nn as nn

# this file will be replaced
 
XBAR_COL_SIZE = 128
XBAR_ROW_SIZE = 128


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
    weight_frac = weight.remainder_(1).mul_(2**n_bit)
    weight = torch.cat([weight_int, weight_frac])
    
    return weight

# fix bit_slicing to 2 bit --> later make it variable
def bit_slice(weight, frac_bit, device):  # version 2

    #assume positive
    # weight.shape[0] is output channels
    # weight.shape[1] is flattened weight length

    int_bit = 15-frac_bit
    # clipping
    max_weight = torch.ones(weight.shape).mul_(2**int_bit-1/2**frac_bit).to(device)
    min_weight = torch.ones(weight.shape).mul_(-2**int_bit).to(device)
    weight = torch.where(weight < (2**int_bit-1/2**frac_bit), weight, max_weight)
    weight = torch.where(weight > -2**int_bit, weight, min_weight)

    out_channel = weight.shape[0]

    weight.mul_(2**(frac_bit-8))  # 0000. 0000 0000 0000 --> 0000 0000. 0000 0000
    weight = slicing(weight,8)
    # ------ 8-bit slice

    weight.div_(2**4)
    weight = slicing(weight,4)
    # ------ 4-bit slice

    weight.div_(2**2)
    weight = slicing(weight,2)
    # ------ 2-bit slice
    
    weight[:out_channel].add_(4).fmod_(4) # for negative numbers. 4 = 2**2. 
    weight[-out_channel:]= torch.floor(weight[-out_channel:])   # last layer

    # already made 2-bit. -> stop.
    # If I use 2^n bit-slice, I d(on't have to slice more to make 1-bits and then combine it again. 

    weight_idx = get_index_rearrange(idx8, out_channel)
    bit_slice = weight[weight_idx[0],:].t()   

    return bit_slice

def float_to_16bits_tensor(input, frac_bit, device): # input is batch x n tensor / output is n x 16 tensor.. version 2
    #assume positive
    # input.shape[0] = batch size
    # input.shape[1] = flattened input length

    int_bit = 15-frac_bit
    # clipping
    max_input = torch.ones(input.shape).mul_(2**int_bit-1/2**frac_bit).to(device)
    min_input = torch.ones(input.shape).mul_(-2**int_bit).to(device)
    input = torch.where(input < (2**int_bit-1/2**frac_bit), input, max_input)
    input = torch.where(input > -2**int_bit, input, min_input)

    batch_size = input.shape[0] 
    input.mul_(2**(frac_bit-8))  # 0000. 0000 0000 0000 --> 0000 0000. 0000 0000
    input = slicing(input,8)
    # ------ 8-bit slice

    input.div_(2**4)
    input = slicing(input,4)
    # ------ 4-bit slice

    input.div_(2**2)
    input = slicing(input,2)
    # ------ 2-bit slice

    input.div_(2)
    input = slicing(input,1)    
    # ------ 1-bit slice
    
    input[0].abs_() # for negative numbers.  
    input[-1]= torch.floor(input[-1])   # last layer
    
    input_idx = get_index_rearrange(idx16, batch_size)
    bit_slice = input[input_idx[0]].reshape(batch_size,16,-1).transpose(1,2)

    return bit_slice



def mvm_tensor(flatten_input, xbars, device):   # version 2

    # xbars shape:          [xbars_row, xbars_col, XBAR_ROW_SIZE, XBAR_COL_SIZE]
    # flatten_input shape:  [batch_size, xbars_row, 128, 16]
    # 2-bit bit-slicing
    xbars_row = xbars.shape[0]
    xbars_col = xbars.shape[1]
    batch_size = flatten_input.shape[0]

    shift_add_1bit = torch.zeros(16) # input bits = 16
    for i in range(16):
        shift_add_1bit[i] = 2**i
    shift_add_1bit[-1] *= -1        # last bit --> subtract
    shift_add_1bit = shift_add_1bit.expand((batch_size, xbars_row, xbars_col, 16, int(XBAR_COL_SIZE/8))).transpose(3,4).to(device)
    
    shift_add_2bit = torch.zeros(8) # 16bit / 2bit-slice
    for i in range(8):
        shift_add_2bit[-i-1] = 4**i
    shift_add_2bit = shift_add_2bit.expand((batch_size, xbars_row, xbars_col, int(XBAR_COL_SIZE/8), 8)).to(device)
    
    output_reg = torch.zeros(batch_size, xbars_row, xbars_col, 16, int(XBAR_COL_SIZE/8)).to(device)

    for i in range(int(XBAR_COL_SIZE/8)):
        input_1bit = flatten_input[:,:,:,-1-i].reshape((batch_size, xbars_row, 1, 128, 1))
        output_analog = torch.sum(torch.mul(xbars, input_1bit), 3).reshape(shift_add_2bit.shape)
        output_reg[:,:,:,i,:] = torch.sum(torch.mul(output_analog, shift_add_2bit), 4)
    output = torch.sum(torch.mul(output_reg, shift_add_1bit), 3)
    output.div_(2**12).trunc_().fmod_(2**16).div_(2**12)

    # + sum xbar_rows
    output = torch.sum(output, 1).reshape(batch_size, -1)
    
    return output



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




