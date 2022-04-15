# -*- coding: utf-8 -*-
'''
Prepare multi-cue data.
'''


__all__ = [
    "PrepareMultiCueDataSamples",
]


import json
import os
import soundfile as sf
import librosa
import random
import numpy as np
from scipy import signal
import yaml
import utils
from random import random as rand
import resampy
import pickle


class PrepareMultiCueDataSamples(object):

    def __init__(self, config):
        # print('Initialize PrepareMultiCueDataSamples...')
        self.config = config
        self.cue_missing_training = config.CUE_MISSING_TRAINING
        self.cue_infering_training = config.CUE_INFERING_TRAINING
        
        self.sample_root_path = config.DATA_PATH
        sample_lst_path = config.DATA_LIST_PATH
        self.train_lst, self.train_num = self.get_sample_lst(
            sample_lst_path[0])
        self.valid_lst, self.valid_num = self.get_sample_lst(
            sample_lst_path[1])
        self.test_lst, self.test_num = self.get_sample_lst(
            sample_lst_path[2])
        self.noise_dB_span = config.NOISE_DB_SPAN
        self.speaker = config.SPEAKER
        self.rever = '_rever' if config.REVER else ''
        self.length = config.MAX_LEN
        self.fps = config.FPS
        self.audio_frame_rate = config.FRAME_RATE
        self.video_length = self.length * self.fps
        self.audio_length = self.length * self.audio_frame_rate
        self.cues = config.CUES
        self.azimuth_noise = config.AZIMUTH_NOISE
        self.silence_rate_threshold = 0.8

    def get_samples(self, phase='train'):
        r"""Defines the data preparation operation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_sample_lst(self, path):
        sample_lst = []
        with open(path, 'r', encoding='utf-8') as fi:
            for sample_line in fi.readlines():
                sample = sample_line.strip().split(' ')
                sample_lst.append(sample)
        sample_num = len(sample_lst)
        return sample_lst, sample_num

    def get_spk_set(self, path):
        spk_set = set()
        if os.path.isdir(path):
            for spk_id in os.listdir(path):
                spk_set.add(spk_id)
        elif os.path.isfile(path):
            with open(path, 'r') as fi:
                for line in fi.readlines():
                    spk_id = line.strip()
                    spk_set.add(spk_id)
        else:
            print("it's a special file(socket, FIFO, device file)")
        all_spk = sorted(spk_set)
        spk_dict = {spk: idx for idx, spk in enumerate(all_spk)}
        # print('Loaded speaker set of %s, with %d spkeakers' %
        #       (path, len(spk_set)))
        return spk_set, spk_dict

    def preprocess_data(self, wav, wav_rate):
        if len(wav.shape) > 1:
            wav = wav[0]  # TODO
        if wav_rate != self.audio_frame_rate:
            wav = resampy.resample(
                wav, wav_rate, self.audio_frame_rate, filter='kaiser_best')
        wav -= np.mean(wav)
        wav /= np.max(np.abs(wav)) + np.spacing(1)
        return wav

    def vad_merge(self, w):
        intervals = librosa.effects.split(w, top_db=20)
        temp = list()
        for s, e in intervals:
            temp.append(w[s:e])
        return np.concatenate(temp, axis=None)

    def get_mel(self, y):
        mel_basis = librosa.filters.mel(sr=8000, n_fft=512, n_mels=40)
        y = librosa.core.stft(y=y, n_fft=512, hop_length=80,
                            win_length=200, window='hann')
        magnitudes = np.abs(y) ** 2
        mel = np.log10(np.dot(mel_basis, magnitudes) + 1e-6)
        return mel

    def get_stft_feas(self, wav, win, stride, feas_type="stft"):
        if feas_type == "stft":
            return np.transpose(np.abs(librosa.core.spectrum.stft(wav, win, stride)))
        elif feas_type == "phase":
            return np.transpose(np.angle(librosa.core.spectrum.stft(wav, win, stride)))
        elif feas_type == "complex":
            return np.transpose(librosa.core.spectrum.stft(wav, win, stride))

    def get_gt_wav_endpoint(self, w, offset=False):
        intervals = librosa.effects.split(w, top_db=20)
        gt_wav_endpoint = np.zeros_like(w)
        for s, e in intervals:
            if offset is False:
                gt_wav_endpoint[s:] = 1
                break
            else:
                # TODO
                # gt_wav_endpoint[s:e] = 1
                gt_wav_endpoint[intervals[0][0]:intervals[-1][-1]] = 1
                gt_wav_intervals = np.array([intervals[0][0], intervals[-1][-1]]) / len(w)
                break
        return gt_wav_endpoint, gt_wav_intervals

    def get_overlap_rate(self, w1, w2):
        assert w1.shape == w2.shape
        overlap = np.zeros_like(w1)
        overlap2 = np.zeros_like(w1)
        for w in (w1, w2):
            intervals = librosa.effects.split(w, top_db=20)
            for s, e in intervals:
                overlap[s:e] += 1
            overlap2[intervals[0][0]:intervals[-1][-1]] += 1
        overlap_rate = np.sum(np.where(overlap == 2, 1, 0)) / \
            np.sum(np.where(overlap > 0, 1, 0))
        overlap_rate2 = np.sum(np.where(overlap2 == 2, 1, 0)) / \
            np.sum(np.where(overlap2 > 0, 1, 0))
        return overlap_rate, overlap_rate2

    def random_shift_data(self, wav):
        random_shift = random.choice(range(len(wav)))
        wav = wav[random_shift:] + wav[:random_shift]
        return wav

    def audio_trim(self, data, start_time=0):
        # channel T
        if len(data.shape) > 1:
            if len(data[0]) > self.audio_length:
                # TODO
                data = data[:, start_time:(start_time+self.audio_length)]
                # guarantee the existence of onset/offset
                # if self.offset is True:
                #     random_cut = random.choice(
                #         range(int(self.audio_frame_rate * 2.2), int(self.audio_frame_rate * 2.9)))
                #     data = data[:random_cut]
                #     data.extend(np.zeros(audio_length - len(data)))
                # else:
                #     data = data[:self.audio_length]
            else:
                data = np.pad(
                    data, ((0, 0), (0, self.audio_length - len(data[0]))), 'constant')
        else:
            if len(data) > self.audio_length:
                data = data[start_time:(start_time+self.audio_length)]
            else:
                data = np.pad(
                    data, ((0, self.audio_length - len(data)),), 'constant')
        return data

    def video_trim(self, data, start_time=0):
        if len(data) > self.video_length:
            data = data[start_time:(start_time+self.video_length)]
            # guarantee the existence of onset/offset
            # if self.offset is True:
            #     random_cut = random.choice(
            #         range(int(self.audio_frame_rate * 2.2), int(self.audio_frame_rate * 2.9)))
            #     data = data[:random_cut]
            #     data.extend(np.zeros(audio_length - len(data)))
            # else:
            #     data = data[:self.video_length]
            # TODO
            if len(data) == 74:
                data = np.pad(data, ((0, 1), (0, 0)), 'constant')
        else:
            data = np.pad(
                data, ((0, self.video_length - len(data)), (0, 0)), 'constant')
        return data

    def video_padding(self, data, target_spk_offset, target_spk_end_offset):
        if target_spk_offset > 0 or target_spk_end_offset > 0:
            visual_offset = round(
                target_spk_offset / self.audio_frame_rate * self.fps)
            visual_end_offset = round(
                target_spk_end_offset / self.audio_frame_rate * self.fps)
            data = np.pad(
                data, ((visual_offset, visual_end_offset), (0, 0)), 'constant')
        return data

    def get_available_cue_lst(self, cues):
        missing_cue_num = random.randint(0, 2)
        missing_cue_num = min(missing_cue_num, len(cues) - 1)
        available_cue_lst = random.sample(
            cues, len(cues) - missing_cue_num)
        return available_cue_lst

    def get_infer_cue_dict(self):
        infer_cue_dict = {'mask': [], 'interfere': [], 'replace': []}
        if rand() < 0.6:  # 60%
            mask_type = 'mask'
            max_mask_cue_count = 2
        elif rand() < 0.5:  # 20%
            mask_type = 'interfere'
            max_mask_cue_count = 3
        else:  # 20%
            mask_type = 'replace'
            max_mask_cue_count = 1
        mask_cue_count = random.randint(1, max_mask_cue_count)
        infer_cue_dict[mask_type] = random.sample(['azimuth', 'visual', 'voiceprint'], mask_cue_count)
        return infer_cue_dict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.load(open(path, 'r')))


if __name__ == "__main__":
    config = read_config('config.yaml')
    samples = PrepareMultiCueDataSamples(config)
    samples_iterator = samples.get_samples(phase='train')
    print('samples:', next(samples_iterator))
