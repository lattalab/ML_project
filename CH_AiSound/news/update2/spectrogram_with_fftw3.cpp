#include <iostream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <sndfile.hh>  // For reading audio files
#include <samplerate.h> // resample
#include <gnuplot-iostream.h> // For plotting
#include <cstring> // For memset
#include <algorithm> // for max function
#include "melfilter.h"

#define SAMPLE_RATE 16000
#define duration  5       // 5 second
#define spec_width  256   // time axis
#define WindowSize  2048  // = n_fft
#define N_FFT 2048
#define N_MELS  128     // freq axis
#define FMAX  SAMPLE_RATE/2   // max frequency
int hopSize = int((SAMPLE_RATE*duration)/(spec_width-1));

// Function to apply a Hamming window
void hamming(int window_Size, float* window) {
    for (int i = 0; i < window_Size; ++i) {
        window[i] = 0.54 - 0.46 * cos(2 * M_PI * i / (window_Size - 1));
    }
}

// Function to read audio file
std::vector<float> readAudioFile(const std::string& filename , int sr=22050) {
    SndfileHandle fileHandle(filename);
    if (fileHandle.error()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return {};
    }
    
    if (fileHandle.channels() != 1) {
        std::cerr << "Error: the audio file is not mono (single channel)" << std::endl;
        return {};
    }

    // Read the audio data
    std::vector<float> audioData(fileHandle.frames());
    fileHandle.read(&audioData[0], fileHandle.frames());

    // Resample the audio data
    double src_ratio = static_cast<double>(sr) / fileHandle.samplerate();
    int outputSampleCount = static_cast<int>(audioData.size() * src_ratio);

    std::vector<float> resampledData(outputSampleCount);
    SRC_STATE *src_state = src_new(SRC_SINC_BEST_QUALITY, 1, NULL);
    if (src_state == NULL) {
        std::cerr << "Error: could not create SRC_STATE" << std::endl;
        return {};
    }

    SRC_DATA src_data;
    src_data.data_in = audioData.data();
    src_data.input_frames = audioData.size();
    src_data.data_out = resampledData.data();
    src_data.output_frames = outputSampleCount;
    src_data.src_ratio = src_ratio;
    src_data.end_of_input = 0;

    int error = src_process(src_state, &src_data);
    if (error) {
        std::cerr << "Error: resampling failed - " << src_strerror(error) << std::endl;
        src_delete(src_state);
        return {};
    }

    src_delete(src_state);

    // Adjust the size of the resampled data in case the output sample count is different
    resampledData.resize(src_data.output_frames_gen);

    return resampledData;
}

// Function to perform STFT
// STFT function with padding
std::vector<std::vector<double>> STFT(const std::vector<float>& signal, int n_fft, int hop_length, int win_length, bool center) {
    int signalLength = signal.size();

    // Default window length to n_fft if not specified
    if (win_length <= 0) {
        win_length = n_fft;
    }

    // Padding for the signal if needed
    int paddingLength = 0;
    if (center) {
        int start_padding = n_fft / 2;
        int end_padding = std::max(0, n_fft - ((signalLength - n_fft / 2) % hop_length) - n_fft / 2);
        paddingLength = start_padding + end_padding;
    }

    int paddedLength = signalLength + paddingLength;
    std::vector<float> paddedSignal(signal);
    paddedSignal.resize(paddedLength, 0.0f);

    fftw_complex* data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_fft);
    fftw_complex* fft_result = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_fft);
    fftw_plan plan_forward = fftw_plan_dft_1d(n_fft, data, fft_result, FFTW_FORWARD, FFTW_ESTIMATE);

    // Create a Hamming window of appropriate length
    float* window = new float[n_fft];
    hamming(n_fft, window);

    std::vector<std::vector<std::pair<float, float>>> spectrogram;

    int chunkPosition = (center) ? n_fft / 2 : 0;

    while (chunkPosition + win_length <= paddedLength) {
        for (int i = 0; i < win_length; ++i) {
            data[i][0] = paddedSignal[chunkPosition + i] * window[i];
            data[i][1] = 0.0;
        }

        // Pad the remaining data if window size is less than n_fft
        for (int i = win_length; i < n_fft; ++i) {
            data[i][0] = 0.0;
            data[i][1] = 0.0;
        }

        // Perform the FFT on our chunk
        fftw_execute(plan_forward);

        std::vector<std::pair<float, float>> fft_frame;
        for (int i = 0; i <= n_fft / 2; ++i) {
            fft_frame.emplace_back(fft_result[i][0], fft_result[i][1]);
        }
        spectrogram.push_back(fft_frame);

        chunkPosition += hop_length;
    }

    delete[] window;
    fftw_destroy_plan(plan_forward);
    fftw_free(data);
    fftw_free(fft_result);

    // Return abs(stft(y))**2
    std::vector<std::vector<double>> magn_fft(spectrogram.size(), std::vector<double>(spectrogram[0].size(), 0.0));
    for (size_t i = 0; i < spectrogram.size(); ++i) {
        for (size_t j = 0; j < spectrogram[i].size(); ++j) {
            float real = spectrogram[i][j].first;
            float imag = spectrogram[i][j].second;
            float magnitude = std::sqrt(real * real + imag * imag);
            magn_fft[i][j] = magnitude*magnitude;
        }
    }

    return magn_fft;
}

// Function to convert power spectrogram to dB scale
// librosa.amplitude_to_db(spectrogram) = power_to_db(S**2, ref=ref**2, amin=amin**2, top_db=top_db)
std::vector<std::vector<double>> power_to_db(const std::vector<std::vector<double>>& power_spec, float amin = 1e-10, float top_db = 80.0) {
    std::vector<std::vector<double>> db_spec(power_spec.size(), std::vector<double>(power_spec[0].size(), 0.f));
    
    double max_val = -std::numeric_limits<double>::infinity();;

    // Convert power to dB scale and find the max value
    for (size_t i = 0; i < power_spec.size(); ++i) {
        for (size_t j = 0; j < power_spec[i].size(); ++j) {
            double magnitude_db = 10 * std::log10(std::max(power_spec[i][j], double(amin)));
            db_spec[i][j] = magnitude_db;
            if (db_spec[i][j] > max_val) {
                max_val = db_spec[i][j];
            }
        }
    }

    // Apply top_db threshold
    double threshold = max_val - top_db;
    for (size_t i = 0; i < db_spec.size(); ++i) {
        for (size_t j = 0; j < db_spec[i].size(); ++j) {
            db_spec[i][j] = std::max(db_spec[i][j], threshold);
        }
    }

    return db_spec;
}

// // Function to plot the spectrogram using Gnuplot
void plotSpectrogram(const std::vector<std::vector<double>>& spectrogram, int sampleRate, int windowSize, int hopSize, int n_mels) {
    // find that need to be transpose
    auto transpose_spectrogram = transpose(spectrogram);
    Gnuplot gp;

    int numFrames = transpose_spectrogram.size();
    int numBins = transpose_spectrogram[0].size(); // Mel bin 的數量
    double Duration = numFrames * hopSize / static_cast<double>(sampleRate);

    // 設定 x 軸和 y 軸標籤
    gp << "set xlabel 'Time (s)'\n";
    gp << "set ylabel 'Frequency (Hz)'\n";
    gp << "set cblabel 'Magnitude (dB)'\n";
    gp << "set pm3d map\n";
    gp << "set palette defined (0 '#440154', 1 '#3B528B', 2 '#21908C', 3 '#5DC863', 4 '#FDE725')\n";
    gp << "unset surface\n";
    gp << "set xrange [0:" << Duration << "]\n";
    gp << "set yrange [0:" << (sampleRate / 2.0) << "]\n"; // 設置 y 軸範圍為最大頻率
    gp << "set ytics 1024\n";  // 將 y 軸刻度設為合理的間隔，如每 2048 Hz 一個標示
    gp << "splot '-' using 1:2:3 with image\n";

    // 傳遞資料到 Gnuplot
    for (int i = 0; i < numFrames; ++i) {
        double time = i * hopSize / static_cast<double>(sampleRate);
        for (int j = 0; j < numBins; ++j) {
            double frequency = j * (sampleRate / 2.0) / (numBins - 1);
            gp << time << " " << frequency << " " << transpose_spectrogram[i][j] << "\n";
        }
        gp << "\n";
    }
    gp << "e\n";
    gp << "e\n";
}


int main() {
    std::string audioPath = "./p225_001.wav";
    std::vector<float> audio = readAudioFile(audioPath, SAMPLE_RATE);

    if (audio.empty()) {
        std::cerr << "Error: no audio data read" << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> spectrogram = STFT(audio, N_FFT, hopSize, WindowSize , true);

    //std::vector<std::vector<double>> spec = calculateMelSpectrogram(spectrogram, SAMPLE_RATE, WindowSize, hopSize, N_MELS); 
    std::vector<std::vector<double>> mel_basis = mel(SAMPLE_RATE, N_FFT, N_MELS);

    try {
        // 計算矩陣乘法
        std::vector<std::vector<double>> transpose_spectrogram = transpose(spectrogram);
        std::vector<std::vector<double>> mel_spectrogram = matmul(mel_basis, transpose_spectrogram);
        mel_spectrogram = power_to_db(mel_spectrogram);
        plotSpectrogram(mel_spectrogram, SAMPLE_RATE, WindowSize, hopSize, N_MELS);
        // 輸出結果 (此處省略)

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
