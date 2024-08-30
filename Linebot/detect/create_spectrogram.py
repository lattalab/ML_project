import warnings
import os
import matplotlib.pyplot as plt
import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import time
import sys
from math import ceil
from random import choice

# Suppress specific UserWarning from numba
warnings.filterwarnings('ignore')

# modify some parameters
# Audio params
# 這個參數表示音訊的採樣率，即每秒採集的樣本數。
SAMPLE_RATE = 22050 
# duration in second 
# 這個參數表示音訊的持續時間，單位為秒。在這個例子中，音訊的持續時間為 5 秒。
DURATION = 5.0
# 這個參數計算了音訊數據的總長度，以樣本數表示。
# 由於採樣率為 16000 Hz，持續時間為 5 秒，因此音訊數據的總長度為 16000 * 5 = 80000 個樣本。
AUDIO_LEN = int(SAMPLE_RATE * DURATION)

# Spectrogram params
N_MELS = 128     # freq axis
N_FFT = 2048
HOP_LEN = 512    # non-overlap region, which means 1/4 portion overlapping
SPEC_WIDTH = AUDIO_LEN // HOP_LEN + 1  # time axis
FMAX = SAMPLE_RATE//2   # max frequency
SPEC_SHAPE = [N_MELS, SPEC_WIDTH]  # expected output spectrogram shape

# Create a directory to store the spectrogram images
destination_folder = "spec"
source_folder = "./audio/"
threshold = 0.0005

# 利用threshold濾掉能量過小的樣本
def envelope(y, rate, threshold):
    y = pd.Series(y).apply(np.abs)
    # 取移動平均
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    mask = y_mean > threshold   # 條件判斷
    return mask, y_mean

def random_padding(y, sr=SAMPLE_RATE):
    n = len(y)  # 音訊長度
    n_all = int(sr * DURATION)  # 5秒的音訊長度
    n_lack = n_all - n  # 音訊不足的部分
    t = n/SAMPLE_RATE  # 時間
    
    if 0.0 < t and t <= 3.0:    # 在3秒內的音訊，複製音訊到滿5秒
        k = ceil(n_lack / n)  
        y_pad = np.tile(y, k)  
        return y_pad[: n_all]  # 取5秒的音訊
    else:   # 大於3秒則，隨機挑一點，接在原音訊後面補足
        point = choice(range(1, n-n_lack))
        y_pad = y[point : point+n_lack]
        return np.concatenate((y, y_pad))

def process_audio_file(filepath):
    start_time = time.time()    # 計時
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)  # 讀取音檔
    # 去除雜訊
    rn_audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.95, n_fft=N_FFT)
    
    # delete margin in the beginning and end (濾掉強度小)
    mask, _ = envelope(rn_audio, SAMPLE_RATE, threshold)
    # find the first and last point larger than threshold
    start_idx = np.argmax(mask)
    # mask[::-1] reverse the mask , np.argmax find the first point larger than threshold
    # and then reverse it back to the original order by calulating the length of the mask - np.argmax(mask[::-1])
    end_idx = len(mask) - np.argmax(mask[::-1]) 
    audio_trimmed = rn_audio[start_idx:end_idx]
    
    # print("start_idx: ", start_idx, "end_idx: ", end_idx, "Actual END:", len(mask))
    # Split audio into segments of 5 seconds
    # if audio length less than 5 sec then should generate 1 picture.
    num_segments = max(1, int(len(audio_trimmed) / AUDIO_LEN))
    
    for i in range(num_segments):
        # 設定取樣區間
        start_sample = i * AUDIO_LEN
        end_sample = start_sample + AUDIO_LEN
        audio_segment = audio_trimmed[start_sample:end_sample]

        # pad the audio with the original audio or cut the audio
        if len(audio_segment) < AUDIO_LEN:
            length_audio = len(audio_segment)
            repeat_count = (AUDIO_LEN + length_audio - 1) // length_audio  # Calculate the `ceiling` of AUDIO_LEN / length_audio
            audio = np.tile(audio_segment, repeat_count)[:AUDIO_LEN]  # Repeat and cut to the required length
        else:
            audio = audio[:AUDIO_LEN]

        # Generate the mel spectrogram
        spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, fmax=FMAX, n_mels=N_MELS, hop_length=HOP_LEN, n_fft=N_FFT)
        spec = librosa.power_to_db(spec)

        # Plot the mel spectrogram
        fig = librosa.display.specshow(spec, sr=SAMPLE_RATE, hop_length=HOP_LEN, x_axis='time', y_axis='mel', cmap='viridis')
        plt.title(f"Spectrogram Segment {i+1}", fontsize=17)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

         # Save the spectrogram image with a meaningful filename
        segment_filename = f"spec_{os.path.basename(filepath)[:-4]}_segment_{i+1}.png"
        save_filepath = os.path.join(destination_folder, segment_filename)
        plt.savefig(save_filepath)
        # Close the figure to free up resources
        plt.close()
    
    end_time = time.time()
    print(f"Processed {os.path.basename(filepath)} in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    os.makedirs(destination_folder, exist_ok=True)  # 存放spectrogram的資料夾

    print("Remove old images...")
    for i in os.listdir(destination_folder):    # 刪除舊的圖片
        i_path = os.path.join(destination_folder, i)
        os.remove(i_path)

    filename = sys.argv[1]
    print("image filename:", filename)
    filepath = os.path.join(source_folder, filename)    # 從這邊讀取音檔
    process_audio_file(filepath)

    print(f"Spectrogram images saved to {destination_folder}")
