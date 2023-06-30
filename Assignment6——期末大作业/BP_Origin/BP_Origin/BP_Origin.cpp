#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;

// Sigmoid激活函数及其导数
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

void backwardPropagation(const vector<vector<double>>& inputs, const vector<vector<double>>& targets,
    vector<vector<double>>& output_weights, vector<vector<double>>& hidden_weights,
    vector<double>& output_bias, vector<double>& hidden_bias) {
    int inputSize = inputs[0].size();
    int hiddenSize = hidden_weights.size();
    int outputSize = output_weights.size();
    int numSamples = inputs.size();

    // 进行反向传播计算
    for (int i = 0; i < numSamples; ++i) {
        const vector<double>& input = inputs[i];
        const vector<double>& target = targets[i];

        // 前向传播
        vector<double> hidden(hiddenSize);
        for (int j = 0; j < hiddenSize; ++j) {
            double z = 0;
            for (int k = 0; k < inputSize; ++k) {
                z += input[k] * hidden_weights[j][k];
            }
            z += hidden_bias[j];
            hidden[j] = sigmoid(z);
        }

        vector<double> output(outputSize);
        for (int j = 0; j < outputSize; ++j) {
            double z = 0;
            for (int k = 0; k < hiddenSize; ++k) {
                z += hidden[k] * output_weights[j][k];
            }
            z += output_bias[j];
            output[j] = sigmoid(z);
        }

        // 计算输出层和隐藏层的误差
        vector<double> outputError(outputSize);
        vector<double> hiddenError(hiddenSize);
        for (int j = 0; j < outputSize; ++j) {
            outputError[j] = (target[j] - output[j]) * sigmoid_derivative(output[j]);
        }
        for (int j = 0; j < hiddenSize; ++j) {
            double errorSum = 0;
            for (int k = 0; k < outputSize; ++k) {
                errorSum += outputError[k] * output_weights[k][j];
            }
            hiddenError[j] = errorSum * sigmoid_derivative(hidden[j]);
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
    int inputSize = 30000;
    int hiddenSize = 2000;
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
        backwardPropagation(inputs, targets, outputWeights, hiddenWeights, outputBias, hiddenBias);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "训练耗时: " << elapsed.count() << " 秒" << std::endl;

    return 0;
}
