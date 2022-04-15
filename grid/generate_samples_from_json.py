# -*- coding: utf-8 -*-
'''
Prepare multi-cue data of Grid.
'''

import json
import os
import random
import numpy as np
import copy
import math


def cal_speaker_direction(spk_pos, sensors_pos):
    '''
    spk_pos: np.array[x, y, z]
    sensors_pos: np.array[[s0_x, s1_x], [s0_y, s1_y], [s0_z, s1_z]]
    '''
    proj_spk_pos = copy.deepcopy(spk_pos)
    proj_sensors_pos = copy.deepcopy(sensors_pos)
    proj_spk_pos[2] = 0
    proj_sensors_pos[2, :] = 0
    
    sensors_center = 0.5 * (proj_sensors_pos[:, 0] + proj_sensors_pos[:, 1])
    rela_pos_sensor0 = proj_sensors_pos[:, 0] - sensors_center
    rela_pos_spk = proj_spk_pos - sensors_center
    # cosine_azimuth = rela_pos_sensor0.dot(
    #     rela_pos_spk) / (np.linalg.norm(rela_pos_sensor0) * np.linalg.norm(rela_pos_spk))
    azimuth_sensor0 = math.atan2(rela_pos_sensor0[1], rela_pos_sensor0[0])  # math.atan2(y, x) -> [-pi, pi]
    azimuth_sensor0 = azimuth_sensor0 if azimuth_sensor0 > 0 else 2 * np.pi + azimuth_sensor0
    azimuth_spk = math.atan2(rela_pos_spk[1], rela_pos_spk[0])
    azimuth_spk = azimuth_spk if azimuth_spk > 0 else 2 * np.pi + azimuth_spk
    azimuth = azimuth_spk - azimuth_sensor0
    azimuth = azimuth if azimuth > 0 else 2 * np.pi + azimuth
    distance = np.linalg.norm(rela_pos_spk)
    return azimuth, distance


def gen_datalist(speaker, rever, phase):
    RIR_path = '/ssd2/dataset/grid/RIR/{}spk{}/{}/'.format(
        speaker, rever, phase)
    json_file = RIR_path + 'stage4_intermediate_resaudio_{}.json'.format(phase)
    stage5_speech_image_path = RIR_path + \
        'stage5_observation{}/speech_image/grid_{}/'.format(speaker, phase)
    stage5_observation_path = RIR_path + \
        'stage5_observation{}/observation/grid_{}/'.format(speaker, phase)
    new_file = 'mix_{}_spk_voice_multi_channel{}_{}.txt'.format(
        speaker, rever, phase)
    with open(json_file, encoding='utf-8') as jf:
        js = json.load(jf)
    sample_dict = js['datasets']['grid_{}'.format(phase)]
    sample_list = list(sample_dict.keys())
    sample_num = len(sample_dict)
    spk_ref_dir = RIR_path + \
        'stage0_speech_zeromean/resaudio_{}/'.format(phase)
    spk_dict = {}
    for spk in os.listdir(spk_ref_dir):
        if not spk in spk_dict.keys():
            spk_dict[spk] = []
        spk_ref_dir_spk = os.path.join(spk_ref_dir, spk)
        for file in os.listdir(spk_ref_dir_spk):
            spk_ref_dir_spk_file = os.path.join(spk_ref_dir_spk, file)
            spk_dict[spk].append(spk_ref_dir_spk_file)

    visual_root_path = '/ssd2/dataset/grid/lip_fea/{}/'.format(phase)
    visual_dict = {}
    for spk in os.listdir(visual_root_path):
        if not spk in visual_dict.keys():
            visual_dict[spk] = []
            visual_spk_path = os.path.join(visual_root_path, spk)
            for file in os.listdir(visual_spk_path):
                visual_file_path = os.path.join(visual_spk_path, file)
                if '.npy' in visual_file_path:
                    visual_dict[spk].append(visual_file_path)

    gender_dict = dict()
    gender_file = 'gender.txt'
    with open(gender_file, 'r', encoding='utf-8') as fi:
        for line in fi.readlines():
            spk, gender = line.strip().split(" ")
            gender_dict[spk] = gender
    with open(new_file, 'w', encoding='utf-8') as fo:
        for sample_key, sample_value in sample_dict.items():
            # print(sample_value)
            if speaker == 2:
                spk1, spk2 = sample_value['speaker_id']
                spk1_file = stage5_speech_image_path + sample_key + '_S0.wav'
                spk2_file = stage5_speech_image_path + sample_key + '_S1.wav'
            elif speaker == 3:
                spk1, spk2, spk3 = sample_value['speaker_id']
                spk1_file = stage5_speech_image_path + sample_key + '_S0.wav'
                spk2_file = stage5_speech_image_path + sample_key + '_S1.wav'
                spk3_file = stage5_speech_image_path + sample_key + '_S2.wav'
            while True:
                spk1_ref = random.choice(spk_dict[spk1])
                if spk1_ref.strip().split('/')[-1] != spk1_file.strip().split('/')[-1]:
                    break
            while True:
                other_spk = random.choice(list(spk_dict.keys()))
                if other_spk != spk1:
                    other_spk_ref = random.choice(spk_dict[other_spk])
                    break
            while True:
                other_spk = random.choice(list(spk_dict.keys()))
                if other_spk != spk1:
                    other_visual  = random.choice(visual_dict[other_spk])
                    break

            spk_wav = spk1_file.strip().split('_resaudio_{}_'.format(phase))[1]
            spk_wav = '/'.join(spk_wav.split('_'))
            spk1_visual = visual_root_path + spk_wav + '.npy'
            sensors = stage5_observation_path + sample_key + '.wav'
            sources_position = np.array(sample_value['source_position'])
            sensors_position = np.array(sample_value['sensor_position'])
            azimuth, target_spk_dis = cal_speaker_direction(
                sources_position[:, 0], sensors_position)
            interfering_spk_azimuth, interfering_spk_dis = cal_speaker_direction(
                sources_position[:, 1], sensors_position)
            target_spk_gender = gender_dict[spk1]
            interfering_spk_gender = gender_dict[spk2]
            if speaker == 2:
                fo.write(sensors+' '+spk1_file+' '+spk2_file+' '+spk1_ref+' '+other_spk_ref+' '+spk1_visual+' '+other_visual+' '+str(azimuth)+' '+str(target_spk_dis)+' '+str(
                    interfering_spk_azimuth)+' '+str(interfering_spk_dis)+' '+str(target_spk_gender)+' '+str(interfering_spk_gender)+'\n')
            elif speaker == 3:
                fo.write(sensors+' '+spk1_file+' '+spk2_file+' '+spk3_file+' '+spk1_ref+' '+other_spk_ref+' '+spk1_visual+' '+other_visual+' '+
                    str(azimuth)+' '+str(target_spk_dis)+'\n')


if __name__ == "__main__":
    speakers = [2]  # [2, 3]
    revers = ['']  # ['', '_rever']
    phases = ['train', 'valid', 'test']
    for speaker in speakers:
        for rever in revers:
            for phase in phases:
                gen_datalist(speaker, rever, phase)
