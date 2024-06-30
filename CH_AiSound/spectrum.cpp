#include "librosa/librosa.h" // 自製librosa庫
#include <iostream>
#include <vector>
#include <sndfile.hh>  // 用於讀取音訊檔案的庫
#include <opencv2/opencv.hpp>  // 用於處理圖像的庫

// 用於讀取單聲道音訊檔案的函數
std::vector<float> read_audio_file(const std::string& filename) {
    SndfileHandle fileHandle(filename);
    if (fileHandle.error()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return {};
    }
    
    if (fileHandle.channels() != 1) {
        std::cerr << "Error: the audio file is not mono (single channel)" << std::endl;
        return {};
    }

    std::vector<float> audioData(fileHandle.frames());
    fileHandle.read(&audioData[0], fileHandle.frames());
    return audioData;
}

int main()
{
  
  // 替換為你的音訊檔案路徑
    std::string audio_path = "./p225_001.wav";
    
    // 讀取音訊檔案
    std::vector<float> x = read_audio_file(audio_path);
    
    if (x.empty()) {
        std::cerr << "Error: no audio data read" << std::endl;
        return 1;
    }

  int sr = 22050;
  int n_fft = 2048;  // length of the FFT size
  int n_hop = 512;  // number of samples between successive frames
  std::string window = "hann";  //window function. currently only supports 'hann'
  bool center = true;  // same as librosa
  std::string pad_mode = "edge"; // pad mode. support "reflect","symmetric","edge"
  float power = 2.f;  // exponent for the magnitude melspectrogram
  int n_mel = 128; // number of mel bands
  int fmin = 0;  // lowest frequency (in Hz)
  int fmax = 8000;  // highest frequency (in Hz)
  bool norm = true; // ortho-normal dct basis
  int type = 2;  // dct type. currently only supports 'type-II'

  // compute mel spectrogram
  std::vector<std::vector<float>> mels = librosa::Feature::melspectrogram(x, sr, n_fft, n_hop, window, center, pad_mode, power,n_mel, fmin, fmax);

  std::cout << "Mel spectrogram values:" << std::endl;
    for (size_t i = 0; i < mels.size(); ++i) {
        for (size_t j = 0; j < mels[i].size(); ++j) {
            std::cout << "mels[" << i << "][" << j << "] = " << mels[i][j] << std::endl;
        }
    }

  return 0;
}
