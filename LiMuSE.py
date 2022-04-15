#coding:utf-8
import torch
import torch.nn as nn
# from torch.nn.common_types import T
import torch.nn.functional as F
from utils import *
from min_max_quantization import *

class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias 
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x

class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = super().forward(x)
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm2d(dim)

class Conv1D_Q(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, QA_flag=False, ak=8):
        super(Conv1D_Q, self).__init__()
        
        self.QA_flag = QA_flag
        self.ak = ak
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, kernel)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))

        if self.QA_flag:
            x = min_max_quantize(x, self.ak)
        
        output = self.conv1d(x)
        return output

class Conv1D(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x, dim=1)
        return x

class ConvTrans1D(nn.ConvTranspose1d):
 
    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x, dim=1)
        return x

class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=False, causal=False):
        super(DepthConv1d, self).__init__()
        
        self.causal = causal
        self.skip = skip
        
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
          groups=hidden_channel,
          padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = select_norm(norm='cln', dim=hidden_channel, shape=3)
            self.reg2 = select_norm(norm='cln', dim=hidden_channel, shape=3)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:,:,:-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual

class DepthConv1d_Q(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=False, causal=False, QA_flag=False, ak=8):
        super(DepthConv1d_Q, self).__init__()
        
        self.causal = causal
        self.skip = skip
        
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
          groups=hidden_channel,
          padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = select_norm(norm='cln', dim=hidden_channel, shape=3)
            self.reg2 = select_norm(norm='cln', dim=hidden_channel, shape=3)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)
        
        self.QA_flag = QA_flag
        self.ak = ak

    def forward(self, input):
        if self.QA_flag:
            input = min_max_quantize(input, self.ak)

        output = self.reg1(self.nonlinearity1(self.conv1d(input)))

        if self.QA_flag:
            output = min_max_quantize(output, self.ak)
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:,:,:-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))

        if self.QA_flag:
            output = min_max_quantize(output, self.ak)
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual

# GC-equipped TCN
class GC_TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 layer, stack, kernel=3, skip=False, 
                 causal=False, dilated=True, num_group=2):
        super(GC_TCN, self).__init__()

        self.receptive_field = 0
        self.dilated = dilated
        self.num_group = num_group
        
        self.TAC = nn.ModuleList([])
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(input_dim//num_group, hidden_dim//num_group, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal)) 
                else:
                    self.TCN.append(DepthConv1d(input_dim//num_group, hidden_dim//num_group, kernel, dilation=1, padding=1, skip=skip, causal=causal))  
                self.TAC.append(TAC(input_dim//num_group, hidden_dim*3//num_group))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
        #print("Receptive field: {:3d} frames.".format(self.receptive_field))
        # output layer
        self.skip = skip
        
    def forward(self, input):
        
        batch_size, N, L = input.shape
        output = input.view(batch_size, self.num_group, -1, L)
        
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                output = self.TAC[i](output)  
                output = output.view(batch_size*self.num_group, -1, L) 
                residual, skip = self.TCN[i](output)
                output = (output + residual).view(batch_size, self.num_group, -1, L) 
                skip_connection = skip_connection + skip 
        else:
            for i in range(len(self.TCN)):
                output = self.TAC[i](output)
                output = output.view(batch_size*self.num_group, -1, L)
                residual = self.TCN[i](output)
                output = (output + residual).view(batch_size, self.num_group, -1, L) 
          
        output = output.view(batch_size, -1, L)
        
        return output

class GC_TCN_Q(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 layer, stack, kernel=3, skip=False, 
                 causal=False, dilated=True, num_group=2, QA_flag=False, ak=8):
        super(GC_TCN_Q, self).__init__()

        self.receptive_field = 0
        self.dilated = dilated
        self.num_group = num_group
        self.skip = skip
        
        self.QA_flag = QA_flag
        self.ak = ak
        
        self.TAC = nn.ModuleList([])
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d_Q(input_dim//num_group, hidden_dim//num_group, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal, QA_flag=QA_flag, ak=ak)) 
                else:
                    self.TCN.append(DepthConv1d_Q(input_dim//num_group, hidden_dim//num_group, kernel, dilation=1, padding=1, skip=skip, causal=causal, QA_flag=QA_flag, ak=ak))  
                self.TAC.append(TAC_Q(input_dim//num_group, hidden_dim*3//num_group, QA_flag=QA_flag, ak=ak))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
        #print("Receptive field: {:3d} frames.".format(self.receptive_field))
        
    def forward(self, input):
        
        batch_size, N, L = input.shape  # B, context*L, N
        output = input.view(batch_size, self.num_group, -1, L) # B, context, L, N
        
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                output = self.TAC[i](output) 
                output = output.view(batch_size*self.num_group, -1, L)
                residual, skip = self.TCN[i](output) 
                output = (output + residual).view(batch_size, self.num_group, -1, L) 
                skip_connection = skip_connection + skip 
        else:
            for i in range(len(self.TCN)):
                output = self.TAC[i](output) 
                output = output.view(batch_size*self.num_group, -1, L)
                residual = self.TCN[i](output)
                output = (output + residual).view(batch_size, self.num_group, -1, L)

        output = output.view(batch_size, -1, L)  # B, N, L
        
        return output

class TCN_Q(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer, stack, 
                kernel=3, skip=True, causal=False, dilated=True, QA_flag=False, ak=8):
        super(TCN_Q, self).__init__()
        
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = select_norm(norm='cln', dim=input_dim, shape=3)
        
        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated
        self.skip = skip
        
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d_Q(input_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal, QA_flag=QA_flag, ak=ak)) 
                else:
                    self.TCN.append(DepthConv1d_Q(input_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal, QA_flag=QA_flag, ak=ak))   
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)       
        #print("Receptive field: {:3d} frames.".format(self.receptive_field))
        self.QA_flag = QA_flag
        self.ak = ak

    def forward(self, input):
        
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](input)
                input = input + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](input)
                output = input + residual
        
        return output

class LiMuSE(nn.Module):

    def __init__(self,
                 N=128,
                 hidden_dim=256,
                 K=32,
                 E=50,
                 layer=24,
                 num_spks=2,
                 context_size=32,
                 group_size=16,
                 activate="relu",
                 causal=False,
                 QA_flag=False,
                 ak=8):
        super(LiMuSE, self).__init__()
        self.E = E
        self.N = N
        self.hidden_dim = hidden_dim
        self.group_size= group_size
        self.num_spks = num_spks
        self.context_size = context_size
        self.layer = layer
        self.encoder = Conv1D(2, N, K, stride=K // 2, padding=0)
        self.voiceprint_encoder = nn.Conv1d(in_channels=512, out_channels=N, kernel_size=1, stride=1, padding=0)
        self.visual_encoder = nn.Linear(256, N)
        self.Normal_S = select_norm('gln', N, 3)

        # context encoder/decoder
        self.context_enc_1 = GC_TCN_Q(self.N, self.hidden_dim, layer=2, stack=1, kernel=3, skip=False, causal=causal, num_group=self.group_size, QA_flag=QA_flag, ak=ak)
        self.context_dec_1 = GC_TCN_Q(self.N, self.hidden_dim, layer=2, stack=1, kernel=3, skip=False, causal=causal, num_group=self.group_size, QA_flag=QA_flag, ak=ak)
        self.context_enc_2 = GC_TCN_Q(3*self.N, 3*self.hidden_dim, layer=2, stack=1, kernel=3, skip=False, causal=causal, num_group=self.group_size, QA_flag=QA_flag, ak=ak)
        self.context_dec_2 = GC_TCN_Q(3*self.N, 3*self.hidden_dim, layer=2, stack=1, kernel=3, skip=False, causal=causal, num_group=self.group_size, QA_flag=QA_flag, ak=ak)
        
        # Separation block
        self.audio_block = GC_TCN_Q(self.N, self.N*4, layer=6, stack=2, kernel=3, skip=False, causal=causal, num_group=self.group_size, QA_flag=QA_flag, ak=ak)
        self.fusion_block = GC_TCN_Q(3*self.N, self.N*12, layer=6, stack=1, kernel=3, skip=False, causal=causal, num_group=self.group_size, QA_flag=QA_flag, ak=ak)
 
        self.gen_masks = Conv1D_Q(3*N, N, 1, QA_flag=QA_flag, ak=ak)
        self.decoder = ConvTrans1D(N, 1, K, stride=K//2)

        # activation function
        active_f = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=0)
        }
        self.activation_type = activate
        self.activation = active_f[activate]


    def forward(self, mix, aux, visual):
        
        enc_out = self.encoder(mix)     # B x N x T
        batch_size, num_channel, T = enc_out.shape
        aux = aux.transpose(1,2)
        aux = self.voiceprint_encoder(aux)  # B x N x 1
        aux = aux.repeat(1,1,T)     # B x N x T
        visual = self.visual_encoder(visual)       # B x max_video_len x T
        visual = F.interpolate(visual.transpose(1,2), T, mode='linear', align_corners=False)   # B x N x T

        audio = self.Normal_S(enc_out)  # B, N, T
        aux = self.Normal_S(aux)
        visual = self.Normal_S(visual)

        ###########  Part 1  ###########
        # context encoding
        squeeze_block, squeeze_rest = split_feature(audio, self.context_size)  # B, N, context, L   torch.Size([B, 128, 32, 376])
        squeeze_frame = squeeze_block.shape[-1] # L

        squeeze_input = squeeze_block.permute(0,3,1,2).contiguous().view(batch_size*squeeze_frame, self.N,
                                                                        self.context_size)  # B*L, N, context

        squeeze_output = self.context_enc_1(squeeze_input) # B*L, N, context

        squeeze_mean = squeeze_output.mean(2).view(batch_size, squeeze_frame,
                                                   self.N).transpose(1,2).contiguous()  # B, N, L
        # sequence modeling
        feature_output = self.audio_block(squeeze_mean).view(batch_size, -1, squeeze_frame)  # B, N, L
        # context decoding
        feature_output = feature_output.unsqueeze(2) + squeeze_block  # B, N, context, L

        feature_output = feature_output.permute(0,3,1,2).contiguous().view(batch_size*squeeze_frame, self.N,
                                                                           self.context_size)  # B*L, N, context

        unsqueeze_output = self.context_dec_1(feature_output).view(batch_size, squeeze_frame, 
                                                                   self.N, -1)  # B, L, N, context
        unsqueeze_output = unsqueeze_output.permute(0,2,3,1).contiguous()  # B, N, context, L
        unsqueeze_output = merge_feature(unsqueeze_output, squeeze_rest)  # B, N, T
        
        ###########  Fusion  ###########
        feature_fusion = torch.cat((unsqueeze_output, aux, visual), dim=2)
        feature_fusion = feature_fusion.reshape(batch_size, -1, T)

        ###########  Part 2  ###########
        # context encoding
        squeeze_block_2, squeeze_rest_2 = split_feature(feature_fusion, self.context_size)  # B, N, context, L
        squeeze_frame_2 = squeeze_block_2.shape[-1]

        squeeze_input_2 = squeeze_block_2.permute(0,3,1,2).contiguous().view(batch_size*squeeze_frame_2, -1,self.context_size)  # B*L, N, context

        squeeze_output_2 = self.context_enc_2(squeeze_input_2) # B*L, N, context
        squeeze_mean_2 = squeeze_output_2.mean(2).view(batch_size, squeeze_frame_2,
                                                   -1).transpose(1,2).contiguous()  # B, N, L
        # Fusion Block
        fusion_output = self.fusion_block(squeeze_mean_2).view(batch_size, -1, squeeze_frame_2)  # B, 3*N, T

        # context decoding
        fusion_output = fusion_output.unsqueeze(2) + squeeze_block_2  # B, 3N, context, L

        fusion_output = fusion_output.permute(0,3,1,2).contiguous().view(batch_size*squeeze_frame_2, -1,
                                                                        self.context_size)  # B*L, N, context

        unsqueeze_output_2 = self.context_dec_2(fusion_output).view(batch_size, squeeze_frame_2, -1,
                                                                    self.context_size)  # B, L, N, context
        unsqueeze_output_2 = unsqueeze_output_2.permute(0,2,3,1).contiguous()  # B, N, context, L
        unsqueeze_output_2 = merge_feature(unsqueeze_output_2, squeeze_rest_2)  # B, N, T

        # Mask Generation
        masks = self.gen_masks(unsqueeze_output_2)
        mask_output = masks * enc_out  # B, N, T

        # Waveform Decoder
        output = self.decoder(mask_output, squeeze=False)  # B, 1, T_wav
        return output

def check_parameters(net):
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_limuse():
    mix = torch.randn(4,2,48000)
    aux = torch.randn(4, 1, 512)
    visual = torch.randn(4, 75, 256)
    nnet = LiMuSE()
    s = nnet(mix,aux,visual)
    print(str(check_parameters(nnet))+' Mb')
    # print(nnet)


if __name__ == "__main__":
    test_limuse()