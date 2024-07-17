import warnings
import os
import matplotlib.pyplot as plt
import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import time
import sys

# Suppress specific UserWarning from numba
warnings.filterwarnings('ignore')

# Audio params
 # 這個參數表示音訊的採樣率，即每秒採集的樣本數。
 # 在這個例子中，採樣率為 16000 Hz，表示每秒會有 16000 個樣本。
SAMPLE_RATE = 16000 
# duration in second 
# 這個參數表示音訊的持續時間，單位為秒。在這個例子中，音訊的持續時間為 5 秒。
DURATION = 5.0
# 這個參數計算了音訊數據的總長度，以樣本數表示。
# 由於採樣率為 16000 Hz，持續時間為 5 秒，因此音訊數據的總長度為 16000 * 5 = 80000 個樣本。
AUDIO_LEN = int(SAMPLE_RATE * DURATION)

# Spectrogram params
N_MELS = 128     # freq axis
N_FFT = 2048
SPEC_WIDTH = 256    # time axis
HOP_LEN = AUDIO_LEN // (SPEC_WIDTH - 1) # non-overlap region
FMAX = SAMPLE_RATE//2   # max frequency
SPEC_SHAPE = [SPEC_WIDTH, N_MELS]   # output spectrogram shape

# Create a directory to store the spectrogram images
destination_folder = "spec"
source_folder = "./audio/"
segment_duration = 5
threshold = 0.0005
sample_field = segment_duration * SAMPLE_RATE   # used to determine the segment number

# 利用threshold濾掉能量過小的樣本
def envelope(y, rate, threshold):
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    mask = y_mean > threshold   # 條件判斷
    return mask, y_mean

def process_audio_file(filepath):
    start_time = time.time()    # 計時
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)  # 讀取音檔
    # 去除雜訊
    rn_audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.95, n_fft=N_FFT)
    
    # delete margin in the beginning and end
    mask, _ = envelope(rn_audio, SAMPLE_RATE, threshold)
    # find the first and last point larger than threshold
    start_idx = np.argmax(mask)
    end_idx = len(mask) - np.argmax(mask[::-1])
    audio_trimmed = rn_audio[start_idx:end_idx]
    
    # print("start_idx: ", start_idx, "end_idx: ", end_idx, "Actual END:", len(mask))
    # Split audio into segments of 5 seconds
    # if audio length less than 5 sec then should generate 1 picture.
    num_segments = max(1, int(len(audio_trimmed) / sample_field))
    
    for i in range(num_segments):
        start_sample = i * sample_field
        end_sample = min(start_sample + sample_field, len(audio_trimmed))
        audio_segment = audio_trimmed[start_sample:end_sample]

        # Generate the mel spectrogram
        spec = librosa.feature.melspectrogram(y=audio_segment, sr=SAMPLE_RATE, fmax=FMAX, n_mels=N_MELS, hop_length=HOP_LEN, n_fft=N_FFT)
        spec = librosa.power_to_db(spec)

        # Plot the mel spectrogram
        fig = librosa.display.specshow(spec, sr=SAMPLE_RATE, hop_length=HOP_LEN, x_axis='time', y_axis='mel', fmax=FMAX, cmap='viridis')
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

    os.makedirs(destination_folder, exist_ok=True)

    filename = sys.argv[1]
    print("image filename:", filename)
    filepath = os.path.join(source_folder, filename)
    process_audio_file(filepath)

    print(f"Spectrogram images saved to {destination_folder}")
