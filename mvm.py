import numpy as np
import torch
import torch.nn as nn
 

## 16 bit fixed point
##   +---+--------------------+------------------------+
##   | 1 |    3    |                12                 |
##   +---+--------------------+------------------------+
##   sign  integer              fraction
##

def float16_to_bits(number):
    if number >= 16 or number < -16:
        print("fixed-point 16bit number should be in -16 <= x < 16")
        return 0
    bin16 = np.zeros(16)
    negative = False
    if number < 0:
        negative = True
    number = abs(number)

    i = 15
    number *= 2<<11 #7
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

def bits_to_float16(bits):
    
    if bits.shape[0] != 16:
        print("It should be 16 bits")
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

def uint_to_bits(number): # adc produce 9-bits
    
    nbit = 9
    bits = np.zeros(nbit)
    for i in range(nbit):
        bits[nbit-1-i] = number%2
        number //= 2
    return bits


def binary_add(a,b):  # a and b should be array of 0,1
                      # ex) a=[0], b=[1], c=[0,1]
    len_a = a.shape[0]
    len_b = b.shape[0]
    length = max(len_a,len_b)
    output = np.zeros(length)

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
        print("Error: Length should be equal")
        return 0
    output = np.zeros(length)
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


def bit_slice_weight(weights, nbit): # weights --> 16-bit fixed-point --> nbit, nbit%2 => 0
  rows = weights.shape[0]
  cols = weights.shape[1]
  cells_per_weight = int(16/nbit)
  
  bit_sliced_weights = np.zeros((rows, cols*cells_per_weight))
  for i in range(rows):
    for j in range(cols):
      fix16b_weight = float16_to_bits(weights[i,j])
      for n in range(cells_per_weight):
        bit_sliced_weights[i, j*cells_per_weight +n] = bits_to_uint(fix16b_weight[2*n:2*n+2])
  
  return bit_sliced_weights


def adc_shift_n_add(adc_out, nbits): # shift and add 8 x 9bit after adc (unsigned)
    n_cells = adc_out.shape[0]
    adc_bits = adc_out.shape[1]
    
    output_bit_len = int(adc_bits + nbits*(n_cells-1)) #23
    output = np.zeros(output_bit_len)
    output2 = np.zeros(output_bit_len)
    output[0:adc_bits] = adc_out[n_cells-1]
    
    for i in range(1,n_cells):
        
        # shift right nbits
        output[nbits:output_bit_len] = output[0:output_bit_len-nbits]
        for j in range(nbits):
            output[j] = 0#output[0]
        
        # add
        output2[0:9] = adc_out[n_cells-1-i]
        output = binary_add(output, output2)
        
    return output


# 1 input set: fixed-point array [n x 16], 
# 1 weight set: Bitsliced float array [n x 16/nbit] 
# --> output: 1 number
def mvm_simple(input_bits, bit_sliced_weights):  
    
    n_cell = bit_sliced_weights.shape[1]  # 8
    nbits = int(16/n_cell) # 2bit 
    output_analog = np.zeros(n_cell)
    output_digital = np.zeros((n_cell, 9))
    output_register = np.zeros(38)
    temp = np.zeros(38)
    
    for i in range(input_bits.shape[1]):    #16):
        for w_i in range(n_cell):      #8):
            output_analog[w_i] = np.sum(input_bits[:,15-i]*bit_sliced_weights[:,w_i])
            
            # ADC
            output_digital[w_i] = uint_to_bits(output_analog[w_i])
        temp[:23] = adc_shift_n_add(output_digital, nbits)            
        
        # shift
        output_register[1:38] = output_register[0:37]
        
        # add
        if i < input_bits.shape[1]-1:
            output_register = binary_add(output_register, temp)
        else: # subtract (the last input bit)    
            output_register = binary_subtract(output_register, temp)

    return output_register[10:26]


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
sum = accumulate(adcout,2)
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn import init

import math
import numpy as np

class Conv2d_mvm_function(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        #output = F.conv2d(input, weight,  bias, stride, padding, dilation, groups)
        
        """"""
        weight_channels_out = weight.shape[0]
        weight_channels_in = weight.shape[1]
        weight_row = weight.shape[2]
        weight_col = weight.shape[3]

        length = weight_channels_in * weight_row * weight_col
        flatten_weight = weight.reshape((weight_channels_out, length))

        input_channels = input.shape[0]     # weight_channels_in == input_channels
        input_row = input.shape[1]
        input_col = input.shape[2]
        

        output_row = input_row - weight_row + 1
        output_col = input_col - weight_col + 1 
        output = torch.zeros((weight_channels_out, output_row, output_col))

        for i in range(output_row):
            for j in range(output_col):
                flatten_input = input[:,:, i:i+weight_row, j:j+weight_col].flatten()

                for k in range(weight_channels_out):
                    output[k,i,j] = sum(flatten_input*flatten_weight[k])

                    
        """"""
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
            

        return grad_input, grad_weight, grad_bias, None, None, None, None


class _ConvNd_mvm(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, check_grad=False):
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', check_grad=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_mvm, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    #@weak_script_method
    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

            return Conv2d_mvm_function.apply(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else:
            return Conv2d_mvm_function.apply(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)





import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

inputs = torch.tensor([[[[1.,0,1],[2,1,0],[1,2,1]],[[2,3,1],[2,0,1],[4,2,1]],[[3,2,1],[0,2,1],[5,3,2]]]])
labels = torch.tensor([1])
weights = np.array([[[[2.,1],[1,2]],[[4,2],[0,1]],[[1,0],[3,2]]],[[[2.,1],[1,2]],[[3,2],[1,1]],[[1,2],[3,2]]]])
trainloader = [[inputs, labels]]
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform =transforms.Compose([transforms.ToTensor()]))
#trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 2, 2, bias=False)
        self.conv1 = Conv2d_mvm(3,2,2, bias=False)
        self.conv1.weight.data = torch.clone(weights)
        self.conv1.weight.requires_grad = True
        #print(weights)
        self.fc1 = nn.Linear(8, 2, bias=False)
        self.fc1.weight.data.fill_(0.1)



    def forward(self, x):
        x = self.conv1(x)
        print(x)
        x = x.view(-1, 8)
        x = self.fc1(x)
        return x


net = Net()

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)




for i, data in enumerate(trainloader):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    #inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)

    print(outputs)
