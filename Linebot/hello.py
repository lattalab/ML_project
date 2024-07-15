import sys
import requests
import os

def download_audio_file(audio_url, save_path):
    response = requests.get(audio_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Audio file downloaded to {save_path}")
    else:
        print("Failed to download audio file")

def process_audio(audio_path):
    # 在這裡添加你的音檔處理邏輯，例如轉換頻譜圖、模型推論等
    print(f"Processing audio file: {audio_path}")
    # ... your processing code ...

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_audio.py <audio_path> <user_id>")
        sys.exit(1)

    audio_path = sys.argv[1]
    user_id = sys.argv[2]

    # 下載音檔
    download_audio_file(f"https://ml-project-1r0x.onrender.com/download_audio/{user_id}", audio_path)

    # 處理音檔
    process_audio(audio_path)
