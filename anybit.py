#!/usr/bin/env python
# -*- coding: utf-8 -*-
# anybit.py is used to quantize the weight of model.

from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy
import pdb
from utils import params_cluster
import os

def sigmoid_t(x, b=0, t=1):
    """
    The sigmoid function with T for soft quantization function.
    Args:
        x: inputesults 
        b: the bias
        t: the temperature
    Returns:
        y = sigmoid(t(x-b))
    """
    temp = -1 * t * (x - b)
    temp = torch.clamp(temp, min=-20.0, max=20.0)
    return 1.0 / (1.0 + torch.exp(temp))

def step(x, bias):
    """ 
    The step function for ideal quantization function in test stage.
    """
    y = torch.zeros_like(x) 
    mask = torch.gt(x - bias,  0.0)
    y[mask] = 1.0
    return y

class QuaOp(object):
    """
    Quantize weight.
    Args:
        model (list): the model to be quantified.
        QW_biases (list): the bias of quantization function.
                          QW_biases is a list with m*n shape, m is the number of layers,
                          n is the number of sigmoid_t.
        QW_values (list): the list of quantization values, 
                          such as [-1, 0, 1], [-2, -1, 0, 1, 2].
    Returns:
        Quantized model.
    """
    def __init__(self, model, QW_biases=[], QW_values=[], initialize_biases=True, init_linear_bias=0):
        count_targets = 0
        for submodel in model:
            for m in submodel.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
                    count_targets = count_targets + 1
        start_range = 1
        end_range = count_targets
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        self.inited = False
        self.param_num = 0 
        index = 0
        for submodel in model:
            for m in submodel.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose1d):
                    index = index + 1
                    if index in self.bin_range:
                        tmp = m.weight.data.clone()
                        self.param_num += tmp.numel()
                        self.saved_params.append(tmp)
                        self.target_modules.append(m.weight)

        print('target_modules number: ', len(self.target_modules))
        self.QW_values = QW_values
        self.n = len(self.QW_values) - 1
        self.threshold = self.QW_values[-1] * 5 / 4.0
        self.scales = []
        offset = 0.
        for i in range(self.n):
            gap = self.QW_values[i + 1] - self.QW_values[i]
            self.scales.append(gap)
            offset += gap
        self.offset = offset / 2.

        if initialize_biases:
            self.QW_biases = []
            self.clusters = []
            if init_linear_bias:
                self.init_linear_bias()
            else:
                self.init_bias(self.QW_values)
        else:
            print('Use the given bias')
            self.QW_biases = QW_biases
        print('scales', self.scales)
        print('offset', self.offset)

    def forward(self, x, T, quan_bias, train=True):
        if train:
            y = sigmoid_t(x, b=quan_bias[0], t=T)*self.scales[0]
            for j in range(1, self.n):
                y += sigmoid_t(x, b=quan_bias[j], t=T)*self.scales[j]
        else:
            y = step(x, bias=quan_bias[0])*self.scales[0] 
            for j in range(1, self.n):
                y += step(x, bias=quan_bias[j])*self.scales[j]

        y = y - self.offset

        return y

    def backward(self, x, T, quan_bias):
        y_1 = sigmoid_t(x, b=quan_bias[0], t=T)*self.scales[0]
        y_grad = (y_1.mul(self.scales[0] - y_1)).div(self.scales[0])
        for j in range(1, self.n):
            y_temp = sigmoid_t(x, b=quan_bias[j], t=T)*self.scales[j]
            y_grad += (y_temp.mul(self.scales[j] - y_temp)).div(self.scales[j])

        return y_grad

    def init_linear_bias(self):
        print('Initializing linear weight quantization biases')
        interval = 2 /  (self.n - 1)
        biases = numpy.arange(-1,1+interval,interval)
        print(biases)
        for param in self.saved_params:
            self.QW_biases.append(biases)
        self.inited = True

    def init_bias(self, QW_values):
        print('Initializing weight quantization biases')
        for param in self.saved_params:
            biases, clusters = params_cluster(param.detach().cpu().numpy(), QW_values, return_cluster=True)
            self.QW_biases.append(biases)
            self.clusters.append(clusters)
        self.inited = True

    def quantization(self, T, alpha, beta, init, train_phase=True):
        """
        The operation of network quantization.
        Args:
            T: the temperature, a single number. 
            alpha: the scale factor of the output, a list.
            beta: the scale factor of the input, a list. 
            init: a flag represents the first loading of the quantization function.
            train_phase: a flag represents the quantization 
                  operation in the training stage.
        """
        self.save_params()
        self.quantizeConvParams(T, alpha, beta, init, train_phase=train_phase)

    def save_params(self):
        """
        save the float parameters for backward
        """
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def restore_params(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


    def quantizeConvParams(self, T, alpha, beta, init, train_phase):
        """
        quantize the parameters in forward
        """
        T = (T > 2000)*2000 + (T <= 2000)*T

        for index in range(self.num_of_params):
            if init:
                beta[index].data = torch.Tensor([self.threshold / self.target_modules[index].data.abs().max()]).cuda()
                alpha[index].data = torch.reciprocal(beta[index].data)

            x = self.target_modules[index].data.mul(beta[index].data)
            
            y = self.forward(x, T, self.QW_biases[index], train=train_phase)

            self.target_modules[index].data = y.mul(alpha[index].data)


    def updateQuaGradWeight(self, T, alpha, beta, init):
        """
        Calculate the gradients of all the parameters.
        The gradients of model parameters are saved in the [Variable].grad.data.
        Args:
            T: the temperature, a single number. 
            alpha: the scale factor of the output, a list.
            beta: the scale factor of the input, a list. 
            init: a flag represents the first loading of the quantization function.
        Returns:
            alpha_grad: the gradient of alpha.
            beta_grad: the gradient of beta.
        """
        beta_grad = [0.0] * len(beta)
        alpha_grad = [0.0] * len(alpha)
        T = (T > 2000)*2000 + (T <= 2000)*T 
        for index in range(self.num_of_params):
            if init:
                beta[index].data = torch.Tensor([self.threshold / self.target_modules[index].data.abs().max()]).cuda()
                alpha[index].data = torch.reciprocal(beta[index].data)
            x = self.target_modules[index].data.mul(beta[index].data)

            # set T = 1 when train binary model
            # y_grad = self.backward(x, 1, self.QW_biases[index]).mul(T)
            # set T = T when train the other quantization model
            y_grad = self.backward(x, T, self.QW_biases[index]).mul(T)
            
        
            beta_grad[index] = y_grad.mul(self.target_modules[index].data).mul(alpha[index].data).\
                               mul(self.target_modules[index].grad.data).sum()
            alpha_grad[index] = self.forward(x, T, self.QW_biases[index]).\
                                mul(self.target_modules[index].grad.data).sum()

            self.target_modules[index].grad.data = y_grad.mul(beta[index].data).mul(alpha[index].data).\
                                                   mul(self.target_modules[index].grad.data)
        return alpha_grad, beta_grad