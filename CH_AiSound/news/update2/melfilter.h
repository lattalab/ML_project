#ifndef MEL
#define MEL

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Helper function to generate FFT frequencies
std::vector<double> fft_frequencies(double sr, int n_fft) {
    std::vector<double> freqs(n_fft / 2 + 1);
    for (int i = 0; i <= n_fft / 2; ++i) {
        freqs[i] = i * sr / n_fft;
    }
    return freqs;
}

// Helper function to convert frequency to Mel scale
double hz_to_mel(double freq, bool htk = false) {
    if (htk) {
        return 2595.0 * std::log10(1.0 + freq / 700.0);
    } else {
        const double f_min = 0.0;
        const double f_sp = 200.0 / 3;
        double mels = (freq - f_min) / f_sp;
        const double min_log_hz = 1000.0;
        const double min_log_mel = (min_log_hz - f_min) / f_sp;
        const double logstep = std::log(6.4) / 27.0;

        if (freq >= min_log_hz) {
            mels = min_log_mel + std::log(freq / min_log_hz) / logstep;
        }

        return mels;
    }
}

// Helper function to convert Mel scale to frequency
double mel_to_hz(double mel, bool htk = false) {
    if (htk) {
        return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    } else {
        const double f_min = 0.0;
        const double f_sp = 200.0 / 3;
        double freqs = f_min + f_sp * mel;
        const double min_log_hz = 1000.0;
        const double min_log_mel = (min_log_hz - f_min) / f_sp;
        const double logstep = std::log(6.4) / 27.0;

        if (mel >= min_log_mel) {
            freqs = min_log_hz * std::exp(logstep * (mel - min_log_mel));
        }

        return freqs;
    }
}

// Helper function to generate Mel frequencies
std::vector<double> mel_frequencies(int n_mels, double fmin, double fmax, bool htk) {
    std::vector<double> mels(n_mels);
    double min_mel = hz_to_mel(fmin, htk);
    double max_mel = hz_to_mel(fmax, htk);
    double mel_step = (max_mel - min_mel) / (n_mels - 1);

    for (int i = 0; i < n_mels; ++i) {
        mels[i] = mel_to_hz(min_mel + i * mel_step, htk);
    }

    return mels;
}

// Function to normalize filter weights
void normalize(std::vector<std::vector<double>>& weights, const std::string& norm) {
    if (norm == "slaney") {
        for (size_t i = 0; i < weights.size(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < weights[i].size(); ++j) {
                sum += weights[i][j];
            }
            if (sum != 0.0) {
                for (size_t j = 0; j < weights[i].size(); ++j) {
                    weights[i][j] /= sum;
                }
            }
        }
    } else {
        throw std::invalid_argument("Unsupported norm=" + norm);
    }
}

// Function to create Mel filter-bank
std::vector<std::vector<double>> mel(double sr, int n_fft, int n_mels = 128, double fmin = 0.0,
                                     double fmax = -1.0, bool htk = false, const std::string& norm = "slaney") {
    if (fmax <= 0.0) {
        fmax = sr / 2.0;
    }

    std::vector<std::vector<double>> weights(n_mels, std::vector<double>(1 + n_fft / 2, 0.0));
    std::vector<double> fftfreqs = fft_frequencies(sr, n_fft);
    std::vector<double> mel_f = mel_frequencies(n_mels + 2, fmin, fmax, htk);
    std::vector<double> fdiff(mel_f.size() - 1);

    for (size_t i = 0; i < fdiff.size(); ++i) {
        fdiff[i] = mel_f[i + 1] - mel_f[i];
    }

    std::vector<std::vector<double>> ramps(n_mels + 2, std::vector<double>(fftfreqs.size()));

    for (size_t i = 0; i < ramps.size(); ++i) {
        for (size_t j = 0; j < fftfreqs.size(); ++j) {
            ramps[i][j] = mel_f[i] - fftfreqs[j];
        }
    }

    for (int i = 0; i < n_mels; ++i) {
        for (size_t j = 0; j < fftfreqs.size(); ++j) {
            double lower = -ramps[i][j] / fdiff[i];
            double upper = ramps[i + 2][j] / fdiff[i + 1];
            weights[i][j] = std::max(0.0, std::min(lower, upper));
        }
    }

    if (!norm.empty()) {
        normalize(weights, norm);
    }

    return weights;
}

// 矩陣轉置
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

// implement np.dot(S, mel_basis)
std::vector<std::vector<double>> matmul(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B
) {
    size_t A_rows = A.size();
    size_t A_cols = A[0].size();
    size_t B_rows = B.size();
    size_t B_cols = B[0].size();

    // 確保矩陣乘法是合法的
    if (A_cols != B_rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    // 初始化結果矩陣
    std::vector<std::vector<double>> C(A_rows, std::vector<double>(B_cols, 0.0));

    // 矩陣乘法計算
    for (size_t i = 0; i < A_rows; ++i) {
        for (size_t j = 0; j < B_cols; ++j) {
            for (size_t k = 0; k < A_cols; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

#endif