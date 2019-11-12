# mvm

Python version: 3.7.3
PyTorch version: 1.1.0

pytorch_sample.py has a simple example inputs and weights matrix, and a simple network.
By the command 'python pytorch_sample.py', you can run the code. 

Conv2d_mvm is the custom module defined in pytorch_mvm_class.py

mvm class has parameters:                                       default value
bit_slice       : # of bit-slicing for weight (1, 2, 4, 8)            2
bit_stream      : # of bit-stream for input (1, 2, 4, 8)              1
weight_bits     : # of bits for fixed point of weight (16, 32)       16
weight_bit_frac : # of bits for fraction part of weight          16 -> 12 / 32 -> 24
input_bits      : # of bits for fixed point of input (16, 32)        16
input_bit_frac  : # of bits for fraction part of input           16 -> 12 / 32 -> 24
adc_bit         : # of adc bits (1, 2, ... )                          9
acm_bits        : # of bit of output                                 16
acm_bit_frac    : # of bits for fraction partof output           16 -> 12 / 32 -> 24


To Indranil & Mustafa

run your code with '-i' command like:
"python pytorch_sample -i"
And then for loops in mvm_tensor will work. 
