import logging
import os
import os.path as osp
import errno
import csv
import codecs
import yaml
import time
import numpy as np
import shutil
import soundfile as sf
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.signal import hilbert
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from min_max_quantization import *
from bss_source import bss_eval_sources

def get_logger(name, format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format='%Y-%m-%d %H:%M:%S', file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_config(path):
    return AttrDict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))

def enframe(signal, nw, inc):
    signal_length = len(signal)
    if signal_length <= nw:
        nf = 1
    else:
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw)
    pad_signal = np.pad(signal, (0, pad_length - signal_length), 'constant')
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf*inc, inc), (nw, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]
    return frames

def get_envelope_wavs(source_wav, wav_rate, win_time=0.032, hop_time=0.008, low_len=1.0):
    win_len = int(win_time * wav_rate)
    hop_len = int(hop_time * wav_rate)
    # T_envelope = np.mean(enframe(np.abs(hilbert(source_wav)), win_len, hop_len), axis=1)
    # T_envelope = np.mean(enframe(np.abs(hilbert(np.concatenate((np.zeros(win_len), source_wav, np.zeros(win_len))))), win_len, hop_len), axis=1)
    nsample = len(source_wav)
    rest = win_len - nsample % hop_len
    # rest = win_len - (hop_len + nsample % win_len) % win_len
    padded_source_wav = np.concatenate((np.zeros(hop_len), source_wav, np.zeros(hop_len), np.zeros(rest)))
    T_envelope = np.mean(enframe(np.abs(hilbert(padded_source_wav)), win_len, hop_len), axis=1)
    # plt.figure()
    # plt.plot(np.arange(len(T_envelope)), T_envelope, linewidth=3)
    # plt.show()
    return T_envelope

# transform-average-concatenate (TAC)
class TAC(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TAC, self).__init__()
        
        self.TAC_input = nn.Sequential(nn.Linear(input_size, hidden_size),
                                       nn.PReLU()
                                      )
        self.TAC_mean = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.PReLU()
                                     )
        self.TAC_output = nn.Sequential(nn.Linear(hidden_size*2, input_size),
                                        nn.PReLU()
                                       )
        self.TAC_norm = nn.GroupNorm(1, input_size)
        
    def forward(self, input):
        # input shape: batch, group, N, seq_length
        
        batch_size, G, N, T = input.shape
        output = input
        
        # transform
        group_input = output  # B, G, N, T
        group_input = output.permute(0,3,1,2).contiguous().view(-1, N)  # B*T*G, N
        group_output = self.TAC_input(group_input).view(batch_size, T, G, -1)  # B, T, G, H
        
        # mean pooling
        group_mean = group_output.mean(2).view(batch_size*T, -1)  # B*T, H
        
        # concate
        group_output = group_output.view(batch_size*T, G, -1)  # B*T, G, H
        group_mean = self.TAC_mean(group_mean).unsqueeze(1).expand_as(group_output).contiguous()  # B*T, G, H
        group_output = torch.cat([group_output, group_mean], 2)  # B*T, G, 2H
        group_output = self.TAC_output(group_output.view(-1, group_output.shape[-1]))  # B*T*G, N
        group_output = group_output.view(batch_size, T, G, -1).permute(0,2,3,1).contiguous()  # B, G, N, T
        group_output = self.TAC_norm(group_output.view(batch_size*G, N, T))  # B*G, N, T
        output = output + group_output.view(input.shape)
        
        return output
    
class TAC_Q(nn.Module):
    def __init__(self, input_size, hidden_size, QA_flag=False, ak=8):
        super(TAC_Q, self).__init__()
        
        self.TAC_input = nn.Sequential(nn.Linear(input_size, hidden_size),
                                       nn.PReLU()
                                      )
        self.TAC_mean = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.PReLU()
                                     )
        self.TAC_output = nn.Sequential(nn.Linear(hidden_size*2, input_size),
                                        nn.PReLU()
                                       )
        self.TAC_norm = nn.GroupNorm(1, input_size)
        
        self.QA_flag = QA_flag
        self.ak = ak

        
    def forward(self, input):
        # input shape: batch, group, N, seq_length
        
        batch_size, G, N, T = input.shape
        output = input
        
        # transform
        group_input = output  # B, G, N, T
        group_input = output.permute(0,3,1,2).contiguous().view(-1, N)  # B*T*G, N
        
        if self.QA_flag:
            group_input = min_max_quantize(group_input, self.ak)
        group_output = self.TAC_input(group_input).view(batch_size, T, G, -1)  # B, T, G, H
        
        # mean pooling
        group_mean = group_output.mean(2).view(batch_size*T, -1)  # B*T, H
        
        # concate
        group_output = group_output.view(batch_size*T, G, -1)  # B*T, G, H
        
        if self.QA_flag:
            group_output = min_max_quantize(group_output, self.ak)
        group_mean = self.TAC_mean(group_mean).unsqueeze(1).expand_as(group_output).contiguous()  # B*T, G, H
        group_output = torch.cat([group_output, group_mean], 2)  # B*T, G, 2H
        
        if self.QA_flag:
            group_output = min_max_quantize(group_output, self.ak)        
        group_output = self.TAC_output(group_output.view(-1, group_output.shape[-1]))  # B*T*G, N
        group_output = group_output.view(batch_size, T, G, -1).permute(0,2,3,1).contiguous()  # B, G, N, T
        group_output = self.TAC_norm(group_output.view(batch_size*G, N, T))  # B*G, N, T
        output = output + group_output.view(input.shape)
        
        return output 

def pad_segment(input, block_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    rest = block_size - (block_stride + seq_len % block_size) % block_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type()).to(input.device)
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, block_stride)).type(input.type()).to(input.device)
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest

def split_feature(input, block_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = pad_segment(input, block_size)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    block1 = input[:,:,:-block_stride].contiguous().view(batch_size, dim, -1, block_size)
    block2 = input[:,:,block_stride:].contiguous().view(batch_size, dim, -1, block_size)
    block = torch.cat([block1, block2], 3).view(batch_size, dim, -1, block_size).transpose(2, 3)

    return block.contiguous(), rest

def merge_feature(input, rest):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, block_size, _ = input.shape
    block_stride = block_size // 2
    input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, block_size*2)  # B, N, K, L

    input1 = input[:,:,:,:block_size].contiguous().view(batch_size, dim, -1)[:,:,block_stride:]
    input2 = input[:,:,:,block_size:].contiguous().view(batch_size, dim, -1)[:,:,:-block_stride]

    output = input1 + input2
    if rest > 0:
        output = output[:,:,:-rest]

    return output.contiguous()  # B, N, T

def params_cluster(params, Q_values, return_cluster=False):
    # print("The max and min values of params: ", params.max(), params.min())
    # print("The shape of params: ", params.shape)

    max_value = abs(params).max().tolist()
    # print("max_abs_value: ", max_value)

    quan_values = Q_values
    threshold = quan_values[-1]*5/4.0
    # print("scale threshold: ", threshold)
    pre_params = np.sort(params.reshape(-1, 1), axis = 0)
    pre_params = pre_params* (threshold/max_value)
    # print('shape of pre_params', pre_params.shape)

    #  cluster 
    n_clusters = len(quan_values)
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(pre_params)
    label_pred = estimator.labels_ 
    centroids = estimator.cluster_centers_ 

    # print("cluster_centers: ", centroids)
    # print("label_pred: ", label_pred)

    temp = label_pred[0]
    saved_index = [0]*(n_clusters - 1)
    j = 0
    for index, i in enumerate(label_pred):
        if i != temp:
            saved_index[j] = index
            j += 1
            temp = i
            
    # print("boundary_index: ", saved_index)

    # print(pre_params[saved_index[0]-1], pre_params[saved_index[0]])
    # print(pre_params[saved_index[1]-1], pre_params[saved_index[1]])

    boundary = [0]*(n_clusters - 1)
    for i in range(n_clusters - 1):
        temp = (pre_params[saved_index[i] - 1] + pre_params[saved_index[i]]) / 2
        boundary[i] = temp.tolist()[0]
    # print("boundary: ", boundary)
    if not return_cluster:
        return boundary
    else:
        return boundary, centroids

def cal_using_wav(batch_size, mix_speech, aim_speech, pre_speech, permutation=False):
    # bs * steps
    SDR_sum = np.array([])
    SDRi_sum = np.array([])
    for idx in range(batch_size):
        pre_speech_channel = pre_speech[idx]
        aim_speech_channel = aim_speech[idx]
        mix_speech_channel = mix_speech[idx]
        aim_speech_channel = np.array(aim_speech_channel.cpu().data)
        pre_speech_channel = np.array(pre_speech_channel.cpu().data)
        mix_speech_channel = np.array(mix_speech_channel.cpu().data)

        result = bss_eval_sources(
            aim_speech_channel, pre_speech_channel, compute_permutation=permutation)
        # print(result)
        SDR_sum = np.append(SDR_sum, result[0])

        SDRi = result[0] - bss_eval_sources(aim_speech_channel,
                                            mix_speech_channel, compute_permutation=permutation)[0]
        # print('SDRi:', SDRi)
        SDRi_sum = np.append(SDRi_sum, SDRi)


    # print('SDR_Aver for this batch:', SDR_sum.mean())
    # print('SDRi_Aver for this batch:', SDRi_sum.mean())
    return SDR_sum.mean(), SDRi_sum.mean()

if __name__ == "__main__":
    pass