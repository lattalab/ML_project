import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

# 設定音檔所在的資料夾路徑
audio_folder = 'D:/CFAD/CFAD/clean_version/train_clean/real_clean/thchs30'
output_folder = 'D:/Project/CH_AiSound/real'

# 確保輸出資料夾存在
# 如果存在就不報錯
os.makedirs(output_folder, exist_ok=True)

# 讀取資料夾內的所有音檔
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]

# 打亂音檔順序(如果只要部分音檔，可以用這個方法)
import random
random.shuffle(audio_files)

print("總共有", len(audio_files), "個音檔")

for audio_file in audio_files:
    # 知道第幾次轉換
    print("正在處理第", audio_files.index(audio_file)+1, "個音檔")

    # 完整的音檔路徑
    file_path = os.path.join(audio_folder, audio_file)

    # 讀取音檔
    y, sr = librosa.load(file_path)

    # 轉換為Mel頻譜圖
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # 轉換為分貝（dB）
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 繪製Mel頻譜圖
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

    # 儲存頻譜圖
    output_file_path = os.path.join(output_folder, audio_file.replace('.wav', '.png'))
    plt.savefig(output_file_path)
    plt.close()

print("所有音檔已轉換為 Mel 頻譜圖並儲存。")
