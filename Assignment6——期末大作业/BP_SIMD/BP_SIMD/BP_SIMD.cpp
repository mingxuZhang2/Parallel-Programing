#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <immintrin.h>

using namespace std;

// Sigmoid激活函数及其导数
__m256d sigmoid(__m256d x) {
    __m256d one = _mm256_set1_pd(1.0);
    __m256d exp_negx = _mm256_exp_pd(_mm256_sub_pd(_mm256_setzero_pd(), x));
    return _mm256_div_pd(one, _mm256_add_pd(one, exp_negx));
}

__m256d sigmoid_derivative(__m256d x) {
    __m256d s = sigmoid(x);
    return _mm256_mul_pd(s, _mm256_sub_pd(_mm256_set1_pd(1.0), s));
}

void backwardPropagationAVX(const vector<vector<double>>& inputs, const vector<vector<double>>& targets,
    vector<vector<double>>& output_weights, vector<vector<double>>& hidden_weights,
    vector<double>& output_bias, vector<double>& hidden_bias) {
    int inputSize = inputs[0].size();
    int hiddenSize = hidden_weights.size();
    int outputSize = output_weights.size();
    int numSamples = inputs.size();

    for (int i = 0; i < numSamples; ++i) {
        const vector<double>& input = inputs[i];
        const vector<double>& target = targets[i];

        // 前向传播
        vector<double> hidden(hiddenSize);
        __m256d hidden_bias_vec = _mm256_loadu_pd(hidden_bias.data());
        for (int j = 0; j < hiddenSize; j += 4) {
            __m256d z = _mm256_setzero_pd();
            for (int k = 0; k < inputSize; ++k) {
                __m256d input_vec = _mm256_set1_pd(input[k]);
                __m256d weight_vec = _mm256_loadu_pd(hidden_weights[j + k].data());
                z = _mm256_fmadd_pd(input_vec, weight_vec, z);
            }
            z = _mm256_add_pd(z, _mm256_broadcast_sd(&hidden_bias[j]));
            _mm256_storeu_pd(&hidden[j], sigmoid(z));
        }

        vector<double> output(outputSize);
        __m256d output_bias_vec = _mm256_loadu_pd(output_bias.data());
        for (int j = 0; j < outputSize; j += 4) {
            __m256d z = _mm256_setzero_pd();
            for (int k = 0; k < hiddenSize; ++k) {
                __m256d hidden_vec = _mm256_set1_pd(hidden[k]);
                __m256d weight_vec = _mm256_loadu_pd(output_weights[j + k].data());
                z = _mm256_fmadd_pd(hidden_vec, weight_vec, z);
            }
            z = _mm256_add_pd(z, _mm256_broadcast_sd(&output_bias[j]));
            _mm256_storeu_pd(&output[j], sigmoid(z));
        }

        // 计算输出层和隐藏层的误差
        vector<double> outputError(outputSize);
        vector<double> hiddenError(hiddenSize);
        for (int j = 0; j < outputSize; ++j) {
            __m256d target_vec = _mm256_set1_pd(target[j]);
            __m256d output_vec = _mm256_loadu_pd(&output[j]);
            __m256d outputError_vec = _mm256_mul_pd(_mm256_sub_pd(target_vec, output_vec), sigmoid_derivative(output_vec));
            _mm256_storeu_pd(&outputError[j], outputError_vec);
        }

        for (int j = 0; j < hiddenSize; ++j) {
            double errorSum = 0;
            for (int k = 0; k < outputSize; ++k) {
                errorSum += outputError[k] * output_weights[k][j];
            }
            for (int j = 0; j < hiddenSize; ++j) {
                __m256d errorSum_vec = _mm256_set1_pd(errorSum);
                __m256d hidden_vec = _mm256_set1_pd(hidden[j]);
                __m256d hiddenError_vec = _mm256_mul_pd(errorSum_vec, sigmoid_derivative(hidden_vec));
                _mm256_storeu_pd(&hiddenError[j], hiddenError_vec);
            }

        }

        // 更新权重和偏置
        for (int j = 0; j < outputSize; ++j) {
            for (int k = 0; k < hiddenSize; ++k) {
                output_weights[j][k] += outputError[j] * hidden[k];
            }
            output_bias[j] += outputError[j];
        }
        for (int j = 0; j < hiddenSize; ++j) {
            for (int k = 0; k < inputSize; ++k) {
                hidden_weights[j][k] += hiddenError[j] * input[k];
            }
            hidden_bias[j] += hiddenError[j];
        }
    }
}

int main() {
    int inputSize = 300000;
    int hiddenSize = 25600;
    int outputSize = 2;
    int numSamples = 100;

    // 使用随机数生成器生成训练数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);

    vector<vector<double>> inputs(numSamples, vector<double>(inputSize));
    vector<vector<double>> targets(numSamples, vector<double>(outputSize));

    for (auto& sample : inputs) {
        for (auto& elem : sample) {
            elem = dis(gen);
        }
    }

    for (auto& sample : targets) {
        for (auto& elem : sample) {
            elem = dis(gen);
        }
    }

    // 初始化权重和偏置
    vector<vector<double>> outputWeights(outputSize, vector<double>(hiddenSize));
    vector<vector<double>> hiddenWeights(hiddenSize, vector<double>(inputSize));
    vector<double> outputBias(outputSize);
    vector<double> hiddenBias(hiddenSize);

    for (auto& row : outputWeights) {
        for (auto& elem : row) {
            elem = dis(gen);
        }
    }

    for (auto& row : hiddenWeights) {
        for (auto& elem : row) {
            elem = dis(gen);
        }
    }

    for (auto& elem : outputBias) {
        elem = dis(gen);
    }

    for (auto& elem : hiddenBias) {
        elem = dis(gen);
    }

    // 训练神经网络
    int epochs = 10;

    auto start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        backwardPropagationAVX(inputs, targets, outputWeights, hiddenWeights, outputBias, hiddenBias);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "训练耗时: " << elapsed.count() << " 秒" << std::endl;

    return 0;
}
