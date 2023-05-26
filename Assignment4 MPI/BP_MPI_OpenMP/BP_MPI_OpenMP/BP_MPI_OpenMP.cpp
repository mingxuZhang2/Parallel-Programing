#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include <mpi.h>

using namespace std;

// Sigmoid激活函数及其导数
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int input_size = 30000;
    int hidden_size = 2000;
    int output_size = 2;
    int num_samples = 100;
    int num_threads = 4;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);

    vector<vector<double>> inputs(num_samples, vector<double>(input_size));
    vector<vector<double>> targets(num_samples, vector<double>(output_size));

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

    vector<vector<double>> output_weights(output_size, vector<double>(hidden_size));
    vector<vector<double>> hidden_weights(hidden_size, vector<double>(input_size));
    vector<double> output_bias(output_size);
    vector<double> hidden_bias(hidden_size);

    for (auto& row : output_weights) {
        for (auto& elem : row) {
            elem = dis(gen);
        }
    }

    for (auto& row : hidden_weights) {
        for (auto& elem : row) {
            elem = dis(gen);
        }
    }

    for (auto& elem : output_bias) {
        elem = dis(gen);
    }

    for (auto& elem : hidden_bias) {
        elem = dis(gen);
    }

    int epochs = 10;
    double learning_rate = 0.1;

    auto start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
#pragma omp parallel num_threads(num_threads) shared(inputs, targets, output_weights, hidden_weights, output_bias, hidden_bias)
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            int chunk_size = num_samples / (worldSize * num_threads);
            int start = rank * num_threads * chunk_size;
            int end = (rank == worldSize - 1) ? num_samples : (rank + 1) * num_threads * chunk_size;

            for (int i = start; i < end; ++i) {
                const vector<double>& input = inputs[i];
                const vector<double>& target = targets[i];

                vector<double> hidden(hidden_size);
                for (int j = 0; j < hidden_size; ++j) {
                    double z = 0;
                    for (int k = 0; k < input_size; ++k) {
                        z += input[k] * hidden_weights[j][k];

                    }
                    z += hidden_bias[j];
                    hidden[j] = sigmoid(z);
                }
                vector<double> output(output_size);
                for (int j = 0; j < output_size; ++j) {
                    double z = 0;
                    for (int k = 0; k < hidden_size; ++k) {
                        z += hidden[k] * output_weights[j][k];
                    }
                    z += output_bias[j];
                    output[j] = sigmoid(z);
                }

                vector<double> output_error(output_size);
                vector<double> hidden_error(hidden_size);
                for (int j = 0; j < output_size; ++j) {
                    output_error[j] = (target[j] - output[j]) * sigmoid_derivative(output[j]);
                }
                for (int j = 0; j < hidden_size; ++j) {
                    double error_sum = 0;
                    for (int k = 0; k < output_size; ++k) {
                        error_sum += output_error[k] * output_weights[k][j];
                    }
                    hidden_error[j] = error_sum * sigmoid_derivative(hidden[j]);
                }

                for (int j = 0; j < output_size; ++j) {
                    for (int k = 0; k < hidden_size; ++k) {
                        output_weights[j][k] += learning_rate * output_error[j] * hidden[k];
                    }
                    output_bias[j] += learning_rate * output_error[j];
                }
                for (int j = 0; j < hidden_size; ++j) {
                    for (int k = 0; k < input_size; ++k) {
                        hidden_weights[j][k] += learning_rate * hidden_error[j] * input[k];
                    }
                    hidden_bias[j] += learning_rate * hidden_error[j];
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, output_weights.data(), output_weights.size() * hidden_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, hidden_weights.data(), hidden_weights.size() * input_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, output_bias.data(), output_bias.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, hidden_bias.data(), hidden_bias.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (rank == 0) {
        std::cout << "训练耗时: " << elapsed.count() << " 秒" << std::endl;
    }

    MPI_Finalize();

    return 0;
}