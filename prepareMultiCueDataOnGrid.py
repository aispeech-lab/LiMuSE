# -*- coding: utf-8 -*-
'''
Prepare multi-cue data of Grid.
'''

__all__ = [
    "PrepareMultiCueGridDataSamples",
]


import json
import os
import soundfile as sf
import random
import numpy as np
from scipy import signal
import yaml
import utils
from random import random as rand
import resampy
import pickle
import copy
# from pyannote.core import SlidingWindow
import torch
from prepareMultiCueDataForGeneral import PrepareMultiCueDataSamples


class PrepareMultiCueGridDataSamples(PrepareMultiCueDataSamples):

    def __init__(self, config):
        super(PrepareMultiCueGridDataSamples, self).__init__(config)
        # print('Initialize PrepareMultiCueGridDataSamples...')
        with open('/home/dataset/grid/grid_vp.pkl', 'rb') as fi:
            self.vp_dict = pickle.load(fi)
        self.previous_batch_num = 0

    def get_samples(self, phase='train'):
        if phase == 'train':
            spk_set, spk_dict = self.spk_train_set, self.spk_train_dict
            sample_lst, sample_num = self.train_lst, self.train_num
            batch_size = self.config.BATCH_SIZE
            previous_batch_num = self.previous_batch_num
        elif phase == 'valid':
            spk_set, spk_dict = self.spk_valid_set, self.spk_valid_dict
            sample_lst, sample_num = self.valid_lst, self.valid_num
            sample_num = min(200, sample_num)
            batch_size = self.config.BATCH_SIZE
            previous_batch_num = 0
        elif phase == 'test':
            spk_set, spk_dict = self.spk_test_set, self.spk_test_dict
            sample_lst, sample_num = self.test_lst, self.test_num
            batch_size = self.config.TEST_BATCH_SIZE
            previous_batch_num = 0
        batch_num = int(sample_num / batch_size)
        # evaluation_interval = min(5000, batch_num-previous_batch_num) if phase == 'train' else batch_num  # TODO
        evaluation_interval = batch_num
        last_batch_num = previous_batch_num + evaluation_interval
        for batch_idx in range(previous_batch_num, last_batch_num + 1):
            if batch_idx == last_batch_num:
                if phase == 'train':
                    self.previous_batch_num = batch_idx if batch_idx < batch_num else 0
                yield False

            # print('\n')
            mix_wav_lst = list()
            spk1_wav_lst = list()
            spk2_wav_lst = list()
            gt_spk1_visual_lst = list()
            gt_spk1_ref_vp_lst = list()

            available_cue_lst = self.cues
            infer_cue_dict = {'mask': [], 'interfere': [], 'replace': []}
            if phase == 'train':
                if self.cue_missing_training:
                    available_cue_lst = self.get_available_cue_lst(self.cues)
                if self.cue_infering_training:
                    infer_cue_dict = self.get_infer_cue_dict()
            # print("available_cue_lst:", available_cue_lst)
            # print("infer_cue_dict:", infer_cue_dict)            

            for batch_sample_idx in range(batch_size):
                sample_idx = batch_idx * batch_size + batch_sample_idx
                sensors_file, spk1_file, spk2_file, spk1_ref_file, other_ref_file, spk1_visual_file, other_visual_file, azimuth, target_spk_dis, interfering_spk_azimuth, interfering_spk_dis, target_spk_gender, interfering_spk_gender = sample_lst[
                    sample_idx]
                sensors_file = os.path.join(self.sample_root_path, sensors_file)
                spk1_file = os.path.join(self.sample_root_path, spk1_file)
                spk2_file = os.path.join(self.sample_root_path, spk2_file)
                spk1_ref_file = os.path.join(self.sample_root_path, spk1_ref_file)
                other_ref_file = os.path.join(self.sample_root_path, other_ref_file)
                spk1_visual_file = os.path.join(self.sample_root_path, spk1_visual_file)
                other_visual_file = os.path.join(self.sample_root_path, other_visual_file)
                mix_wav, wav_rate = sf.read(sensors_file)
                mix_wav = mix_wav.transpose()  # channel * T
                if not 'binaural' in available_cue_lst:
                    mix_wav[1] = mix_wav[0]
                    azimuth = 0
                spk1_wav, wav_rate = sf.read(spk1_file)
                spk1_wav = spk1_wav.transpose()[0]
                spk2_wav, wav_rate = sf.read(spk2_file)
                spk2_wav = spk2_wav.transpose()[0]

                spk1_ref_wav, wav_rate = sf.read(spk1_ref_file)
                spk1_ref_wav = spk1_ref_wav.transpose()
                spk1_ref_wav = self.preprocess_data(spk1_ref_wav, wav_rate)
                other_ref_wav, wav_rate = sf.read(other_ref_file)
                other_ref_wav = other_ref_wav.transpose()
                other_ref_wav = self.preprocess_data(other_ref_wav, wav_rate)                
                # spk1_ref_vp = self.vp_dict['_'.join(spk1_ref_file.split('/')[-2:])]
                spk1_ref_vp = self.vp_dict['_'.join(spk1_file.split('/')[-1].split('_')[4:6])+'.wav']  # 目标纯净语音声纹做GT声纹
                other_ref_vp = self.vp_dict['_'.join(other_ref_file.split('/')[-2:])]

                # spk_wav = spk1_file.strip().split(
                #     '_resaudio_{}_'.format(phase))[1]
                # spk_wav = '/'.join(spk_wav.split('_'))
                try:
                    spk1_visual = np.load(spk1_visual_file)
                except FileNotFoundError as e:
                    # print(e)
                    spk1_visual = np.zeros((75, 256))
                    visual_valid = 0
                else:
                    visual_valid = 1
                try:
                    other_spk_visual = np.load(other_visual_file)
                except FileNotFoundError as e:
                    # print(e)
                    other_spk_visual = np.zeros((75, 256))
                    
                azimuth = float(azimuth)
                target_spk_dis = float(target_spk_dis)
                interfering_spk_azimuth = float(interfering_spk_azimuth)
                interfering_spk_dis = float(interfering_spk_dis)
 
                # interfere和replace在线索输入层面实现
                gt_azimuth, gt_spk1_visual, gt_spk1_ref_vp, gt_spk1_ref_wav = copy.deepcopy(azimuth), copy.deepcopy(spk1_visual), copy.deepcopy(spk1_ref_vp), copy.deepcopy(spk1_ref_wav)
                azimuth, spk1_visual, spk1_ref_vp, spk1_ref_wav = self.get_masked_cues(infer_cue_dict, azimuth, spk1_visual, other_spk_visual, spk1_ref_vp, other_ref_vp, spk1_ref_wav, other_ref_wav)

                mix_wav_lst.append(mix_wav)
                spk1_wav_lst.append(spk1_wav)
                spk2_wav_lst.append(spk2_wav)
                
                gt_spk1_visual_lst.append(gt_spk1_visual)

                gt_spk1_ref_vp_lst.append(gt_spk1_ref_vp)

            mix_wav_lst = torch.tensor(mix_wav_lst, dtype=torch.float)                 
            spk1_wav_lst = torch.tensor(spk1_wav_lst, dtype=torch.float)
            spk2_wav_lst = torch.tensor(spk2_wav_lst, dtype=torch.float)
            gt_spk1_ref_vp_lst = torch.tensor(gt_spk1_ref_vp_lst, dtype=torch.float)
            gt_spk1_visual_lst = torch.tensor(gt_spk1_visual_lst, dtype=torch.float)
            ref_wav_lst = [spk1_wav_lst, spk2_wav_lst]
            yield {'mix_wav': mix_wav_lst,
                   'spk1_wav': spk1_wav_lst,
                   'ref_wav': ref_wav_lst,
                   'gt_visual': gt_spk1_visual_lst,
                   'gt_ref_vp': gt_spk1_ref_vp_lst}

if __name__ == "__main__":
    config = utils.read_config('config.yml')
    grid_samples = PrepareMultiCueGridDataSamples(config)
    
    epoch = 0
    while epoch < 5:
        epoch += 1
        step = 0
        grid_samples_iterator = grid_samples.get_samples(phase='valid')
        while True:
            val_data = grid_samples_iterator.__next__()
            print(val_data['mix_wav'])
            print(val_data['gt_ref_vp'])
            print(val_data['gt_visual'])
            print(val_data['spk1_wav'])
            print(val_data['ref_wav'])
            if val_data == False:
                break
            print('step:{}', step)
            step += 1