import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn import Parameter
import math

import numpy as np
import os
import matplotlib.pyplot as plt


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def min_max_quantize(x, k):
    n = 2 ** k
    a = torch.min(x)
    b = torch.max(x)
    s = (b - a) / (n - 1)

    x = torch.clamp(x, float(a), float(b))
    x = (x - a) / s
    x = RoundFunction.apply(x)
    x = x * s + a
    return x

def min_max_quantize2(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    min_val, max_val = input.min(), input.max()

    input_rescale = (input - min_val) / (max_val - min_val)

    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n

    v =  v * (max_val - min_val) + min_val
    return v

if __name__ == "__main__":
    x = torch.Tensor([-4, 0.222 ,0.5, 0.489, 11, 1])
    b = min_max_quantize(x, 2)
    print(b)