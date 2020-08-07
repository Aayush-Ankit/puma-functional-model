## Profile runtime of spare matrix mul compared to dense-fp32 (simd cores) and dense-fp16 (tensor cores) 

import sys
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse

torch.backends.cudnn.benchmark = True

def densemm(A, B):
    return torch.matmul(A, B)

def densemm_fp16(A, B):
    return torch.matmul(A.half(), B.half())

def sparsemm(A, B):
    return torch.sparse.mm(A, B)

# Setup parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--M', default=32, type=int, 
    help='first gemm dimension')
parser.add_argument('-n', '--N', default=32, type=int, 
    help='second gemm dimension')
parser.add_argument('-k', '--K', default=32, type=int, 
    help='third gemm dimension')
parser.add_argument('--stdout', action='store', default='None', 
    help='path to dump the results')
args = parser.parse_args()

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
#device = "cpu"
print ("Device available: ", device)

# Setup operands for GEMM: M,N,K -D = A*B + C
A =  torch.randn(args.M, args.K).to(device)
B =  torch.randn(args.K, args.N).to(device)

A_h  = A.half()
B_h  = B.half()

warmup_iter  = 10
profile_iter = 100

# Profile dense-fp32 operation
for i in range (warmup_iter):
    densemm(A, B)

torch.cuda.synchronize()
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range (profile_iter):
        densemm(A, B)
print(prof.key_averages())

# Profile dense-fp16 operation
for i in range (warmup_iter):
    densemm_fp16(A_h, B_h)

torch.cuda.synchronize()
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range (profile_iter):
        densemm_fp16(A_h, B_h)
print(prof.key_averages())

# Profile sparse operation
sparse_l = [0.5, 0.75, 0.875]

for s in  sparse_l:
    A_temp = A.clone()
    if (int(s*A.nelement()) > 0):
        topk = torch.topk(torch.abs(A).view(-1), k=int(s*A.nelement()), largest=False)
        A_temp.view(-1)[topk.indices] = 0.0    
    A_s = A_temp.to_sparse()

    for i in range (warmup_iter):
        sparsemm(A_s, B)

    torch.cuda.synchronize()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range (profile_iter):
            sparsemm(A_s, B)
    print(prof.key_averages())
