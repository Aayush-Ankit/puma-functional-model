import numpy as np
import torch
import torch.nn as nn
 

## 16 bit fixed point
##   +---+--------------------+------------------------+
##   | 1 |    3    |                12                 |
##   +---+--------------------+------------------------+
##   sign  integer              fraction
##

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
    return bits


def binary_add(a,b):  # a and b should be array of 0,1
                      # ex) a=[0], b=[1], c=[0,1]
    len_a = a.shape[0]
    len_b = b.shape[0]
    length = max(len_a,len_b)
    output = torch.zeros(length)

    c = 0
    for i in range(min(len_a,len_b)):
        if a[len_a-i-1]+b[len_b-i-1]+c ==3:
            output[length-i-1] = 1
            c = 1
        elif a[len_a-i-1]+b[len_b-i-1]+c == 2:
            output[length-i-1] = 0
            c = 1
        elif a[len_a-i-1]+b[len_b-i-1]+c == 1:
            output[length-i-1] = 1
            c = 0
        else:
            output[length-i-1] = 0
            c = 0
    #output[0] += c
    return output
  
def binary_subtract(a,b):  #a-b
  
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
    output2 = torch.zeros(output_bit_len)

    output[:adc_bits] = adc_out[n_cells-1]
    for i in range(1,n_cells):
        
        # shift right nbits
        output[nbits:output_bit_len] = output[:output_bit_len-nbits]
        for j in range(nbits):
            output[j] = 0
        
        # add
        output2[:adc_bits] = adc_out[n_cells-1-i]
        output = binary_add(output, output2)
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
    temp = torch.zeros(38)
    for i in range(input_bits.shape[1]):    #16):
        #print(i)
        for w_i in range(n_cell):      #8):
            output_analog[w_i] = torch.sum(input_bits[:,15-i]*bit_sliced_weights[:,w_i])
            
            # ADC
            output_digital[w_i] = uint_to_bits(output_analog[w_i])
        temp[:23] = adc_shift_n_add(output_digital, nbits)            
        
        # shift
        output_register[1:] = output_register[:-1]
        
        # add
        if i < input_bits.shape[1]-1:
            output_register = binary_add(output_register, temp)
        else: # subtract (the last input bit)    
            output_register = binary_subtract(output_register, temp)

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




"""
adcout = np.zeros((8,9))
adcout[7] = np.array([0,0,0,1,1,0,0,1,1])
adcout[6] = np.array([1,0,1,0,1,0,0,1,1])
adcout[5] = np.array([1,0,0,1,1,0,1,0,0])
adcout[4] = np.array([0,0,1,0,1,1,0,0,0])

a1 = bits_to_uint(adcout[7])
a2 = bits_to_uint(adcout[6])
a3 = bits_to_uint(adcout[5])
a4 = bits_to_uint(adcout[4])

print(a1, a2, a3, a4)
print(a1+4*a2+16*a3+64*a4)
sum = adc_shift_n_add(adcout,2)
print(bits_to_uint(sum))

#a=fix16(1.52)
#b=fix16(-1.251)
#w_a = fix16(2.252)
#w_b = fix16(-1.51)
bit_a = float16_to_bits(1.52)
bit_b = float16_to_bits(-1.251)
input = np.array([bit_a,bit_b])
weight = np.array([[2.252], [1.25]])
bit_weight=bit_slice_weight(weight,2)
print(input)
print(bit_weight)
print(1.52*2.252-1.251*1.25)
a=mvm_simple(input, bit_weight)
print(bits_to_float16(a))


"""

