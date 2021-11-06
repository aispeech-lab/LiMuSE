# coding=utf8
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
import librosa
from sklearn import metrics
from scipy.signal import hilbert

import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.load(open(path, 'r')))


def read_datas(filename, trans_to_num=False):
    lines = open(filename, 'r').readlines()
    lines = list(map(lambda x: x.split(), lines))
    if trans_to_num:
        lines = [list(map(int, line)) for line in lines]
    return lines


def save_datas(data, filename, trans_to_str=False):
    if trans_to_str:
        data = [list(map(str, line)) for line in data]
    lines = list(map(lambda x: " ".join(x), data))
    with open(filename, 'w') as f:
        f.write("\n".join(lines))


def logging(file):
    def write_log(s, end='\n'):
        print(s, end=end)
        with open(file, 'a') as f:
            f.write(s)

    return write_log


def logging_csv(file):
    def write_csv(s):
        # with open(file, 'a', newline='') as f:
        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(s)

    return write_csv


def format_time(t):
    return time.strftime("%Y-%m-%d-%H:%M:%S", t)


def eval_metrics(reference, candidate, label_dict, log_path):
    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)
    ref_file = ref_dir + 'reference'
    cand_file = cand_dir + 'candidate'

    for i in range(len(reference)):
        with codecs.open(ref_file + str(i), 'w', 'utf-8') as f:
            f.write("".join(reference[i]) + '\n')
        with codecs.open(cand_file + str(i), 'w', 'utf-8') as f:
            f.write("".join(candidate[i]) + '\n')

    def make_label(l, label_dict):
        length = len(label_dict)
        result = np.zeros(length)
        indices = [label_dict.get(label.strip().lower(), 0) for label in l]
        result[indices] = 1
        return result

    def prepare_label(y_list, y_pre_list, label_dict):
        reference = np.array([make_label(y, label_dict) for y in y_list])
        candidate = np.array([make_label(y_pre, label_dict)
                              for y_pre in y_pre_list])
        return reference, candidate

    def get_metrics(y, y_pre):
        hamming_loss = metrics.hamming_loss(y, y_pre)
        macro_f1 = metrics.f1_score(y, y_pre, average='macro')
        macro_precision = metrics.precision_score(y, y_pre, average='macro')
        macro_recall = metrics.recall_score(y, y_pre, average='macro')
        micro_f1 = metrics.f1_score(y, y_pre, average='micro')
        micro_precision = metrics.precision_score(y, y_pre, average='micro')
        micro_recall = metrics.recall_score(y, y_pre, average='micro')
        return hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall

    y, y_pre = prepare_label(reference, candidate, label_dict)
    hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall = get_metrics(y,
                                                                                                                 y_pre)
    return {'hamming_loss': hamming_loss,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall}


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def save_checkpoint(state, is_best, fpath = 'checkpoint.pt'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pt'))

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("==> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("==> No checkpoint found at '{}'".format(fpath))


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes the time something cost"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.begin = time.time()
    
    def cost(self):
        return time.time() - self.begin


def save_separated_speech(train_data, idx, savefrom, saveto):
    if not os.path.exists(saveto):
        os.makedirs(saveto)
    for filename in os.listdir(savefrom):
        new_filename = '{}_{}'.format(idx, filename)
        srcFile = os.path.join(savefrom, filename)
        targetFile = os.path.join(saveto, new_filename)
        shutil.copyfile(srcFile, targetFile)


def save_samples(config, predict_wav, train_data, batch_idx, dst):
    target_spk_clean_wav = train_data['spk1_wav'][0]
    batch_idx_pre = batch_idx.split('_')[0]
    batch_idx_post = '_'.join(batch_idx.split('_')[1:])
    sf.write(dst + '/{}_clean_{}.wav'.format(batch_idx_pre, batch_idx_post),
             target_spk_clean_wav, config.FRAME_RATE)
    predict_wav = predict_wav.contiguous().view(-1)  # B*spk*T 1*1*T
    sf.write(dst + '/{}_pre_{}.wav'.format(batch_idx_pre, batch_idx_post),
             predict_wav.cpu().numpy(), config.FRAME_RATE)
    sf.write(dst + '/{}_noisy_{}.wav'.format(batch_idx_pre, batch_idx_post),
             np.transpose(train_data['mix_wav'][0]), config.FRAME_RATE)


def enframe(signal, nw, inc):
    signal_length = len(signal)  # 信号总长度
    if signal_length <= nw:  # 如果信号长度小于一帧的长度，则帧数定义为1
        nf = 1  # nf表示帧数量
    else:
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))  # 处理后，所有帧的数量,上取整
    pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的平铺后的长度
    pad_signal = np.pad(signal, (0, pad_length - signal_length), 'constant')  # 0填充最后一帧
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf*inc, inc), (nw, 1)).T  # 每帧的索引
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]  # 得到帧信号, 用索引拿数据
    return frames
    

# def get_envelope_wavs(wav_file_path, win_time=0.040, hop_time=0.040, low_len=1.0):
#     source_wav, wav_rate = sf.read(wav_file_path)
#     if len(source_wav.shape) > 1:  # 这里检查通道数
#         source_wav = source_wav[:, 0]
#     if len(source_wav) < low_len * wav_rate:
#         print('WARNING: TOO SHORT SOUND FILE (LEN:%d): %s' % (len(source_wav), wav_file_path))
#     source_wav -= np.mean(source_wav)  # 追加减均值操作
#     source_wav /= np.max(np.abs(source_wav)) + np.spacing(1)
#     # 画包络
#     win_len = int(win_time * wav_rate)
#     hop_len = int(hop_time * wav_rate)
#     T_envelope = np.mean(enframe(np.abs(hilbert(source_wav)), win_len, hop_len), axis=1)
#     # plt.figure()
#     # plt.plot(np.arange(len(T_envelope)), T_envelope, linewidth=3)
#     # plt.show()
#     return T_envelope


def get_envelope_wavs(source_wav, wav_rate, win_time=0.032, hop_time=0.008, low_len=1.0):
    # 画包络
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