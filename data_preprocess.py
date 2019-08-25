#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import librosa
import numpy as np
from hparam import hparam as hp
import gzip
import tarfile
###thchs30#####
# thchs30_path = '/home/momozyc/Music/audioData/thchs30/data_thchs30/data'
# audio_files = glob.glob(os.path.join(thchs30_path, '*.wav'))
# #print(audio_files)
# n = 0
# audio_speakers = []
# for wav_file in audio_files:
#     wav_list = wav_file.strip().split('/')
#     spkID = wav_list[-1].split('_')[0]
#     if spkID not in audio_speakers:
#         audio_speakers.append(spkID)

# audio_wav_dir = []

# i = 0
# for one in audio_speakers:
#     #print(one)
#     audio_wav_dir.append([])
#     for wav in audio_files:
#         wav_list = wav.strip().split('/')
#         spkID = wav_list[-1].split('_')[0]
#         if spkID == one:
#             audio_wav_dir[int(i)].append(wav)           
#     i += 1                                       
########thchs30#####

#####aidatatang#####
#datatang = '/home/momozyc/Music/speaker-verification-data-chinese-lms/datatang/aidatatang_200zh/corpus'
#wav_dirs = glob.glob(os.path.join(datatang, '*', '*', '*'))
#print(wav_tars)

# def extract(tar_path, target_path):
#     try:
#         tar = tarfile.open(tar_path, "r:gz")
#         file_names = tar.getnames()
#         for file_name in file_names:
#             tar.extract(file_name, target_path)
#         tar.close()
#     except Exception  as e:
#         print(e)

# for file in wav_tars:
#     tar_path = file
#     target_path = tar_path[:-7]
#     extract(tar_path, target_path)
#     #print(target_path)
# extract(tar_path, target_path)
#######aidatatang#####

#######aishell1######
# aishell1_path = '/home/momozyc/Music/speaker-verification-data-chinese-lms/aishell1/wav'
# wav_tars = glob.glob(os.path.join(aishell1_path, "*"))
# print(wav_tars)
# def extract(tar_path, target_path):
#     try:
#         tar = tarfile.open(tar_path, "r:gz")
#         file_names = tar.getnames()
#         for file_name in file_names:
#             tar.extract(file_name, target_path)
#         tar.close()
#     except Exception  as e:
#         print(e)

# for file in wav_tars:
#     tar_path = file
#     target_path = tar_path[:-7]
#     extract(tar_path, target_path)
#     #print(target_path)
# extract(tar_path, target_path)
# wav_dirs = glob.glob(os.path.join(aishell1_path, '*', '*', '*'))
#print(wav_dirs)
#######aishell1######

######ST-cmd######
# st_cmd_path = '/home/momozyc/Music/speaker-verification-data-chinese-lms/st_cmds'
# wav_files = glob.glob(os.path.join(st_cmd_path, '*', '*.wav'))
# wav_dirs = []
# for wav_file in wav_files:
#     wav_dir = wav_file[:-9]
#     if wav_dir not in wav_dirs:
#         wav_dirs.append(wav_dir)

# wav_folders = []
# i = 0
# for wav_dir in wav_dirs:
#     wav_folders.append([])
#     for every in wav_files:
#         if wav_dir in every:
#             wav_folders[i].append(every)
#     i += 1

#print(wav_dirs, len(wav_dirs))
#print(wav_folders, len(wav_folders))

####周星驰###
path = '/home/momozyc/Music/speaker-verification-data-chinese-lms/zhouxingchi_audio'
wav_files = glob.glob(os.path.join(path, '*', '*.wav'))
wav_folders = []
wav_folders.append(wav_files)
print(wav_folders)

def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")
    os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(hp.data.test_path, exist_ok=True)    # make folder to save test file
    os.makedirs(hp.data.zhouxingchi_path, exist_ok=True)
    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    total_speaker_num = len(wav_folders)
    train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("total speaker number : %d"%total_speaker_num)
    print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    for i, folder in enumerate(wav_folders):
        #print(folder)
        print("%dth speaker processing..."%i)
        utterances_spec = []
        #folders = glob.glob(os.path.join(folder, '*.wav'))
        print(folder)
        folders = filter(lambda x:len(librosa.core.load(x, hp.data.sr)[0]) > 20, folder)
        for utter_name in folders:
            #print(utter_name)
            if utter_name[-4:] == '.wav':
                utter_path = utter_name         # path of each utterance
                utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
                print(utter.shape)
                #if utter.shape != 0: 
                intervals = librosa.effects.split(utter, top_db=30)       # voice activity detection
                for interval in intervals:
                        if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                            utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                            S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                            S = np.abs(S) ** 2
                            mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                            S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                            utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
                            utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        #if i<train_speaker_num:      # save spectrogram as numpy file
        np.save(os.path.join(hp.data.zhouxingchi_path, "speaker%d.npy"%(i)), utterances_spec)
        #else:
            #np.save(os.path.join(hp.data.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)


if __name__ == "__main__":
    save_spectrogram_tisv()