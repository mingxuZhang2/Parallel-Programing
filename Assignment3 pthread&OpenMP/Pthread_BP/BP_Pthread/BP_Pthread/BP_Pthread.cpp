#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <pthread.h>
#pragma comment(lib, "pthreadVC2.lib")
using namespace std;

// Sigmoid激活函数及其导数
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

struct BPData {
    const vector<vector<double>>* inputs;
    const vector<vector<double>>* targets;
    vector<vector<double>>* output_weights;
    vector<vector<double>>* hidden_weights;
    vector<double>* output_bias;
    vector<double>* hidden_bias;
    int start;
    int end;
};

void* bp_thread(void* arg) {
    BPData* data = static_cast<BPData*>(arg);
    const vector<vector<double>>& inputs = *data->inputs;
    const vector<vector<double>>& targets = *data->targets;
    vector<vector<double>>& output_weights = *data->output_weights;
    vector<vector<double>>& hidden_weights = *data->hidden_weights;
    vector<double>& output_bias = *data->output_bias;
    vector<double>& hidden_bias = *data->hidden_bias;
    int start = data->start;
    int end = data->end;

    double learning_rate = 0.1;

    for (int i = start; i < end; ++i) {
        const vector<double>& input = inputs[i];
        const vector<double>& target = targets[i];

        int input_size = input.size();
        int hidden_size = hidden_weights.size();
        int output_size = output_weights.size();

        // 前向传播
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

        // 计算输出层和隐藏层的误差
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
        // 更新权重和偏置
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

    return nullptr;
}

int main() {
    int input_size = 30000;
    int hidden_size = 2000;
    int output_size = 2;
    int num_samples = 100;
    int num_threads = 4;// 使用随机数生成器生成训练数据
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

    // 初始化权重和偏置
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

    // 训练神经网络
    int epochs = 10;

    vector<pthread_t> threads(num_threads);
    vector<BPData> thread_data(num_threads);

    int chunk_size = num_samples / num_threads;

    auto start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < num_threads; ++i) {
            thread_data[i].inputs = &inputs;
            thread_data[i].targets = &targets;
            thread_data[i].output_weights = &output_weights;
            thread_data[i].hidden_weights = &hidden_weights;
            thread_data[i].output_bias = &output_bias;
            thread_data[i].hidden_bias = &hidden_bias;
            thread_data[i].start = i * chunk_size;
            thread_data[i].end = (i == num_threads - 1) ? num_samples : (i + 1) * chunk_size;

            pthread_create(&threads[i], nullptr, bp_thread, &thread_data[i]);
        }

        for (int i = 0; i < num_threads; ++i) {
            pthread_join(threads[i], nullptr);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "训练耗时: " << elapsed.count() << " 秒" << std::endl;
    return 0;
}
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <random>
//#include <chrono>
//#include <thread>
//#include <mutex>
//#include <condition_variable>
//#include <algorithm>
//
//using namespace std;
//
//// Sigmoid激活函数及其导数
//double sigmoid(double x) {
//    return 1.0 / (1.0 + exp(-x));
//}
//
//double sigmoid_derivative(double x) {
//    double s = sigmoid(x);
//    return s * (1 - s);
//}
//
//struct BPData {
//    const vector<vector<double>>* inputs;
//    const vector<vector<double>>* targets;
//    vector<vector<double>>* output_weights;
//    vector<vector<double>>* hidden_weights;
//    vector<double>* output_bias;
//    vector<double>* hidden_bias;
//    int start;
//    int end;
//    mutex* mtx;
//    condition_variable* cv;
//    bool* finished;
//};
//
//void bp_thread(BPData* data) {
//    const vector<vector<double>>& inputs = *data->inputs;
//    const vector<vector<double>>& targets = *data->targets;
//    vector<vector<double>>& output_weights = *data->output_weights;
//    vector<vector<double>>& hidden_weights = *data->hidden_weights;
//    vector<double>& output_bias = *data->output_bias;
//    vector<double>& hidden_bias = *data->hidden_bias;
//    int start = data->start;
//    int end = data->end;
//
//    double learning_rate = 0.1;
//
//    for (int i = start; i < end; ++i) {
//        const vector<double>& input = inputs[i];
//        const vector<double>& target = targets[i];
//
//        int input_size = input.size();
//        int hidden_size = hidden_weights.size();
//        int output_size = output_weights.size();
//
//        // 前向传播
//        vector<double> hidden(hidden_size);
//        for (int j = 0; j < hidden_size; ++j) {
//            double z = 0;
//            for (int k = 0; k < input_size; ++k) {
//                z += input[k] * hidden_weights[j][k];
//            }
//            z += hidden_bias[j];
//            hidden[j] = sigmoid(z);
//        }
//
//        vector<double> output(output_size);
//        for (int j = 0; j < output_size; ++j) {
//            double z = 0;
//            for (int k = 0; k < hidden_size; ++k) {
//                z += hidden[k] * output_weights[j][k];
//            }
//            z += output_bias[j];
//            output[j] = sigmoid(z);
//        }
//        // 计算输出层和隐藏层的误差
//        vector<double> output_error(output_size);
//        vector<double> hidden_error(hidden_size);
//        for (int j = 0; j < output_size; ++j) {
//            output_error[j] = (target[j] - output[j]) * sigmoid_derivative(output[j]);
//        }
//        for (int j = 0; j < hidden_size; ++j) {
//            double error_sum = 0;
//            for (int k = 0; k < output_size; ++k) {
//                error_sum += output_error[k] * output_weights[k][j];
//            }
//            hidden_error[j] = error_sum * sigmoid_derivative(hidden[j]);
//        }
//
//        // 更新权重和偏置
//        lock_guard<mutex> lock(*data->mtx);
//        for (int j = 0; j < output_size; ++j) {
//            for (int k = 0; k < hidden_size; ++k) {
//                output_weights[j][k] += learning_rate * output_error[j] * hidden[k];
//            }
//            output_bias[j] += learning_rate * output_error[j];
//        }
//        for (int j = 0; j < hidden_size; ++j) {
//            for (int k = 0; k < input_size; ++k) {
//                hidden_weights[j][k] += learning_rate * hidden_error[j] * input[k];
//            }
//            hidden_bias[j] += learning_rate * hidden_error[j];
//        }
//    }
//
//    {
//        lock_guard<mutex> lock(*data->mtx);
//        *data->finished = true;
//    }
//    data->cv->notify_all();
//}
//
//int main() {
//    int input_size = 3;
//    int hidden_size = 4;
//    int output_size = 2;
//    int num_samples = 100;
//    int num_threads = thread::hardware_concurrency();
//    // 使用随机数生成器生成训练数据
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> dis(-1, 1);
//    vector<vector<double>> inputs(num_samples, vector<double>(input_size));
//    vector<vector<double>> targets(num_samples, vector<double>(output_size));
//
//    for (auto& sample : inputs) {
//        for (auto& elem : sample) {
//            elem = dis(gen);
//        }
//    }
//
//    for (auto& sample : targets) {
//        for (auto& elem : sample) {
//            elem = dis(gen);
//        }
//    }
//
//    // 初始化权重和偏置
//    vector<vector<double>> output_weights(output_size, vector<double>(hidden_size));
//    vector<vector<double>> hidden_weights(hidden_size, vector<double>(input_size));
//    vector<double> output_bias(output_size);
//    vector<double> hidden_bias(hidden_size);
//
//    for (auto& row : output_weights) {
//        for (auto& elem : row) {
//            elem = dis(gen);
//        }
//    }
//
//    for (auto& row : hidden_weights) {
//        for (auto& elem : row) {
//            elem = dis(gen);
//        }
//    }
//
//    for (auto& elem : output_bias) {
//        elem = dis(gen);
//    }
//
//    for (auto& elem : hidden_bias) {
//        elem = dis(gen);
//    }
//
//    // 训练神经网络
//    int epochs = 1000;
//    int chunk_size = num_samples / num_threads;
//    vector<thread> threads(num_threads);
//    vector<BPData> thread_data(num_threads);
//    mutex mtx;
//    condition_variable cv;
//    vector<char>thread_finished(num_threads, false);
//    auto start = std::chrono::high_resolution_clock::now();
//
//    for (int epoch = 0; epoch < epochs; ++epoch) {
//        for (int i = 0; i < num_threads; ++i) {
//            thread_data[i].inputs = &inputs;
//            thread_data[i].targets = &targets;
//            thread_data[i].output_weights = &output_weights;
//            thread_data[i].hidden_weights = &hidden_weights;
//            thread_data[i].output_bias = &output_bias;
//            thread_data[i].hidden_bias = &hidden_bias;
//            thread_data[i].start = i * chunk_size;
//            thread_data[i].end = (i == num_threads - 1) ? num_samples : (i + 1) * chunk_size;
//            thread_data[i].mtx = &mtx;
//            thread_data[i].cv = &cv;
//            thread_data[i].finished = &thread_finished[i];
//
//            threads[i] = thread(bp_thread, &thread_data[i]);
//        }
//
//        {
//            unique_lock<mutex> lock(mtx);
//            cv.wait(lock, [&thread_finished] {
//                return all_of(thread_finished.begin(), thread_finished.end(), [](char f) { return f; });
//                });
//            fill(thread_finished.begin(), thread_finished.end(), false);
//        }
//    }
//
//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> elapsed = end - start;
//    std::cout << "训练耗时: " << elapsed.count() << " 秒" << std::endl;
//    return 0;
//
//}