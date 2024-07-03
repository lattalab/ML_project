import os
import matplotlib.pyplot as plt
from observe_audio_function import load_audio, get_mel_spectrogram, plot_mel_spectrogram, envelope, normalize_audio, SAMPLE_RATE, AUDIO_LEN
import warnings
from observe_audio_function import denoise, process_audio

# Suppress specific UserWarning from numba
warnings.filterwarnings("ignore", message="FNV hashing is not implemented in Numba")

# get the constant and functions


# Create a directory to store the spectrogram images
destination_folder = "../spec"
source_folder = "../audio"

if __name__ == "__main__":
    # os.makedirs(destination_folder, exist_ok=True)

    # for filename in os.listdir(source_folder):
        # filepath2 = os.path.join(source_folder, filename)
    fp = './CH_fake_noise.wav'  # 測試部分新增
    audio, sr = load_audio(fp, sr=SAMPLE_RATE)
    audio = audio[:AUDIO_LEN]
    rn = denoise(audio, sr=SAMPLE_RATE) # 新增去雜音
    # delete the low energy part
    # mask, _ = envelope(audio, sr, threshold = 0.0005)  # can adjust the threshold
    # audio = audio[mask]
    spec = get_mel_spectrogram(rn)
    # min-max the spec
    # spec = normalize_audio(spec)
    fig = plot_mel_spectrogram(spec)
    plt.title("Spectrogram", fontsize=17)
    
    plt.show()  # 新增來看輸出的圖

    # Save the spectrogram image with a meaningful filename
    # filename = f"spec_{filename[:-5]}.png"  # Use single quotes inside the f-string
    # filepath = os.path.join(destination_folder, filename)
    # plt.savefig(filepath)

    # Close the figure to free up resources
    plt.close()

    # print(f"Spectrogram images saved to {destination_folder}")
