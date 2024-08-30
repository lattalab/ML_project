import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa 
import pandas as pd
import random
import os
import seaborn as sns

# Audio params
SAMPLE_RATE = 22050  # (samples/sec)
DURATION = 5.0  # duration in second (sec)
AUDIO_LEN = int(SAMPLE_RATE * DURATION)  # total number of samples in DURATION

# Spectrogram params
N_MELS = 128  # freq axis, number of filters
N_FFT = 2048  # frame size
HOP_LEN = 512  # non-overlap region, which means 1/4 portion overlapping
SPEC_WIDTH = AUDIO_LEN // HOP_LEN + 1  # time axis
FMAX = SAMPLE_RATE // 2  # max frequency, based on the rule, it should be half of SAMPLE_RATE
SPEC_SHAPE = [N_MELS, SPEC_WIDTH]  # expected output spectrogram shape

def load_audio(filepath, sr=SAMPLE_RATE):  # load the audio
    audio, sr = librosa.load(filepath, sr=sr)
    return audio, sr

def get_mel_spectrogram_mean(audio, sr=SAMPLE_RATE):  # Get the mel-spectrogram
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, fmax=FMAX, n_mels=N_MELS, hop_length=HOP_LEN, n_fft=N_FFT)
    spec = librosa.power_to_db(spec)  # Turn into log-scale
    spec_mean = np.mean(spec, axis=1)
    return spec_mean

def get_mfcc_mean(audio, sr=SAMPLE_RATE):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, fmax=FMAX, n_mels=N_MELS, hop_length=HOP_LEN, n_fft=N_FFT)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def get_chroma_mean(audio, sr=SAMPLE_RATE):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12, n_fft=N_FFT, hop_length=HOP_LEN)
    # Take the mean of the chroma features along the time axis to get a 1D representation
    chroma_stft_mean = np.mean(chroma, axis=1)  # (12,256)
    return chroma_stft_mean

def RMS_mean(audio, sr=SAMPLE_RATE):
    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LEN)
    return np.mean(rms, axis=1)

def zcr_mean(audio, sr=SAMPLE_RATE):
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=N_FFT, hop_length=HOP_LEN)
    return np.mean(zcr, axis=1)

def spec_centroid_mean(audio, sr=SAMPLE_RATE):
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    return np.mean(spec_centroid, axis=1)

def tonnetz_mean(audio, sr=SAMPLE_RATE):
    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr, hop_length=HOP_LEN)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    return tonnetz_mean

def write_to_csv(audio_source_folder, csv_filename, label, func, val):
    # 讀取資料夾內的所有音檔
    audio_files = [f for f in os.listdir(audio_source_folder) if f.endswith(('.wav', '.mp3', '.flac'))]
    random.shuffle(audio_files)
    audio_files = audio_files[:1000]     # 隨機取前500筆資料

    results = []
    for audio_file in audio_files:
        # 完整的音檔路徑
        file_path = os.path.join(audio_source_folder, audio_file)
        audio, sr = load_audio(file_path, sr=SAMPLE_RATE)
        # 將檔案名與chroma_stft_mean包在一個list中存入results
        mean = func(audio, sr=SAMPLE_RATE).tolist()
        results.append([audio_file] + mean + [label])

    # Convert the results to a pandas DataFrame
    columns = ['Audio_name'] + [f'{val}_{i+1}' for i in range(len(mean))] + ['Label']
    df = pd.DataFrame(results, columns=columns)

    # Check if CSV file exists
    if not os.path.isfile(csv_filename):
        # If file does not exist, write a new CSV file
        df.to_csv(csv_filename, index=False)
    else:
        # If file exists, append new content to the existing CSV file
        df.to_csv(csv_filename, mode='a', header=False, index=False)