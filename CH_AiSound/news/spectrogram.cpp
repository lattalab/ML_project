#include "librosa/librosa.h" // 自製librosa庫
#include <iostream>
#include <vector>
#include <sndfile.hh>  // 用於讀取音訊檔案的庫
#include <gnuplot-iostream.h> // 用於處理圖像的庫
#include <cmath> // power_to_db (10 log S)

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

int getSamplerate(const std::string& filename){
    SndfileHandle fileHandle(filename);
    return fileHandle.samplerate();
}

// Function to convert power spectrogram to dB scale
std::vector<std::vector<float>> power_to_db(const std::vector<std::vector<float>>& power_spec) {
    std::vector<std::vector<float>> db_spec(power_spec.size(), std::vector<float>(power_spec[0].size(), 0.f));

    for (size_t i = 0; i < power_spec.size(); ++i) {
        for (size_t j = 0; j < power_spec[i].size(); ++j) {
            db_spec[i][j] = 10.f * log10(power_spec[i][j]);
        }
    }

    return db_spec;
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

  int sr = getSamplerate(audio_path);
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

  // Convert power spectrogram to dB scale
  std::vector<std::vector<float>> mel_db = power_to_db(mels);

  // Prepare data for Gnuplot
    Gnuplot gp;
    gp << "set xlabel 'Time (frames)'\n";
    gp << "set ylabel 'Frequency (Hz)'\n";
    gp << "set zlabel 'Magnitude (dB)'\n";
    gp << "set pm3d map\n";
    // viridis 
    gp << "set palette defined (0 '#440154', 1 '#3B528B', 2 '#21908C', 3 '#5DC863', 4 '#FDE725')\n";
    gp << "set cblabel 'Magnitude (dB)'\n";
    gp << "unset surface\n";
    gp << "splot '-' matrix with image\n";

    for (size_t i = 0; i < mel_db.size(); ++i) {
        for (size_t j = 0; j < mel_db[i].size(); ++j) {
            gp << mel_db[i][j] << " ";
        }
        gp << "\n";
    }

    gp << "e\n";
    gp << "e\n"; // End of matrix

  return 0;
}
