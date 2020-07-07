import torch.nn as nn
import torch
import torch.nn.functional as F

## Use global parameters (below) for all layers or layer specific parameters
val = True
ifglobal_weight_bits = val
ifglobal_weight_bit_frac = val
ifglobal_input_bits = val
ifglobal_input_bit_frac = val
ifglobal_xbar_col_size = val
ifglobal_xbar_row_size = val
ifglobal_tile_col = val
ifglobal_tile_row = val
ifglobal_bit_stream = val
ifglobal_bit_slice = val
ifglobal_adc_bit = val
ifglobal_acm_bits = val
ifglobal_acm_bit_frac = val
ifglobal_xbmodel = val
ifglobal_xbmodel_weight_path = val

## Fixed point arithmetic configurations
weight_bits = 16
weight_bit_frac = 12
input_bits = 16
input_bit_frac = 12

## Tiling configurations
xbar_col_size = 16
xbar_row_size = 32
tile_col = 2
tile_row = 2

## Bit-slicing configurations
bit_stream = 1
bit_slice = 2
adc_bit = 14
acm_bits = 32
acm_bit_frac = 24

## GENIEx configurations
loop = False # executes GENIEx with batching when set to False

non_ideality = False
class NN_model(nn.Module):
    def __init__(self, N):
         super(NN_model, self).__init__()
         print ("WARNING: crossbar sizes with different row annd column dimension not supported.")
         self.fc1 = nn.Linear(N**2+N, 500)
         # self.bn1 = nn.BatchNorm1d(500)
         self.relu1 = nn.ReLU(inplace=True)
         self.do2 = nn.Dropout(0.5)
         self.fc3 = nn.Linear(500,N)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.do2(out)
        out = self.fc3(out)
        return out

#xbmodel = NN_model(xbar_row_size)
xbmodel = None
#xbmodel_weight_path = 'xb_models/XB_'+str(xb_size)+'_stream1slice2.pth.tar'
xbmodel_weight_path = None

# Dump the current global configurations
def dump_config():
    param_dict = {'weight_bits':weight_bits, 'weight_bit_frac':weight_bit_frac, 'input_bits':input_bits, 'input_bit_frac':input_bit_frac, 
                  'xbar_row_size':xbar_row_size, 'xbar_col_size':xbar_col_size, 'tile_row':tile_row, 'tile_col':tile_col,
                  'bit_stream':bit_stream, 'bit_slice':bit_slice, 'adc_bit':adc_bit, 'acm_bits':acm_bits, 'acm_bit_frac':acm_bit_frac,
                  'non-ideality':ind, 'xbmodel':xbmodel, 'xbmodel_weight_path':xbmodel_weight_path}

    print("********************************************")
    print("Functional simulator configurations:")
    for key, val in param_dict.items():
        print (key, val)
    print("********************************************")