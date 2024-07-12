#include <iostream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <sndfile.hh>  // For reading audio files
#include <gnuplot-iostream.h> // For plotting
#include <cstring> // For memset
#include <algorithm> // for max function

// Function to apply a Hamming window
void hamming(int windowSize, float* window) {
    for (int i = 0; i < windowSize; ++i) {
        window[i] = 0.54 - 0.46 * cos(2 * M_PI * i / (windowSize - 1));
    }
}

// Function to read audio file
std::vector<float> readAudioFile(const std::string& filename) {
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
    fileHandle.read(&audioData[0], fileHandle.frames());    // push data
    return audioData;
}

// Function to perform STFT
std::vector<std::vector<std::pair<float, float>>> STFT(const std::vector<float>& signal, int windowSize, int hopSize) {
    int signalLength = signal.size();
    fftw_complex* data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * windowSize);
    fftw_complex* fft_result = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * windowSize);
    fftw_plan plan_forward = fftw_plan_dft_1d(windowSize, data, fft_result, FFTW_FORWARD, FFTW_ESTIMATE);

     // Create a hamming window of appropriate length
    float window[windowSize];
    hamming(windowSize, window);

    std::vector<std::vector<std::pair<float, float>>> spectrogram;

    int chunkPosition = 0;
    int readIndex;
    bool bStop = false;     // Should we stop reading in chunks? 

     // Process each chunk of the signal
    while (chunkPosition < signalLength && !bStop) {
        for (int i = 0; i < windowSize; ++i) {
            readIndex = chunkPosition + i;
            if (readIndex < signalLength) {
                data[i][0] = signal[readIndex] * window[i];
                data[i][1] = 0.0;
            } else {
                data[i][0] = 0.0;
                data[i][1] = 0.0;
                bStop = true;
            }
        }

        // Perform the FFT on our chunk
        fftw_execute(plan_forward);

        std::vector<std::pair<float, float>> fft_frame;
        for (int i = 0; i <= windowSize / 2; ++i) {
            fft_frame.emplace_back(fft_result[i][0], fft_result[i][1]);
        }
        spectrogram.push_back(fft_frame);

        chunkPosition += hopSize;
    }

    fftw_destroy_plan(plan_forward);
    fftw_free(data);
    fftw_free(fft_result);

    return spectrogram;
}

// Function to convert power spectrogram to dB scale
// librosa.amplitude_to_db(spectrogram) = power_to_db(S**2, ref=ref**2, amin=amin**2, top_db=top_db)
std::vector<std::vector<float>> power_to_db(const std::vector<std::vector<std::pair<float, float>>>& power_spec, float amin = 1e-10, float top_db = 80.0) {
    std::vector<std::vector<float>> db_spec(power_spec.size(), std::vector<float>(power_spec[0].size(), 0.f));
    
    float max_val = -std::numeric_limits<float>::infinity();;

    // Convert power to dB scale and find the max value
    for (size_t i = 0; i < power_spec.size(); ++i) {
        for (size_t j = 0; j < power_spec[i].size(); ++j) {
            float real = power_spec[i][j].first;
            float imag = power_spec[i][j].second;
            float magnitude = std::sqrt(real * real + imag * imag);
            float magnitude_db = 20 * std::log10(std::max(magnitude, float(amin*amin)));
            db_spec[i][j] = magnitude_db;
            if (db_spec[i][j] > max_val) {
                max_val = db_spec[i][j];
            }
        }
    }

    // Apply top_db threshold
    float threshold = max_val - top_db;
    for (size_t i = 0; i < db_spec.size(); ++i) {
        for (size_t j = 0; j < db_spec[i].size(); ++j) {
            db_spec[i][j] = std::max(db_spec[i][j], threshold);
        }
    }

    return db_spec;
}

// Function to plot the spectrogram using Gnuplot
void plotSpectrogram(const std::vector<std::vector<std::pair<float, float>>>& spectrogram, int sampleRate, int windowSize, int hopSize) {
    Gnuplot gp;
    gp << "set xlabel 'Time (s)'\n";
    gp << "set ylabel 'Frequency (Hz)'\n";
    gp << "set zlabel 'Magnitude (dB)'\n";
    gp << "set pm3d map\n";
    gp << "set palette defined (0 '#440154', 1 '#3B528B', 2 '#21908C', 3 '#5DC863', 4 '#FDE725')\n";
    gp << "set cblabel 'Magnitude (dB)'\n";
    gp << "unset surface\n";
    gp << "splot '-' matrix with image\n";

    std::vector<std::vector<float>> log_spectrogram = power_to_db(spectrogram);
    int numFrames = log_spectrogram.size();
    int numBins = windowSize / 2 + 1;

    for (int i = 0; i < numBins; ++i) {
        for (int j = 0; j < numFrames; ++j) {
            gp << log_spectrogram[j][i] << " ";
        }
        gp << "\n";
    }
    gp << "e\n";
    gp << "e\n";
}

int main() {
    std::string audioPath = "./p225_001.wav";
    std::vector<float> audio = readAudioFile(audioPath);

    if (audio.empty()) {
        std::cerr << "Error: no audio data read" << std::endl;
        return 1;
    }

    int sampleRate = 16000; // 16000 hz /s
    int duration = 5;       // 5 second
    int spec_width = 256;   // time axis
    int windowSize = 2048;  // = n_fft
    int hopSize = int((sampleRate*duration)/(spec_width-1));

    std::vector<std::vector<std::pair<float, float>>> spectrogram = STFT(audio, windowSize, hopSize);

    plotSpectrogram(spectrogram, sampleRate, windowSize, hopSize);

    return 0;
}
