#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <random>
using namespace std;

std::vector<float> convolution_avx2(const std::vector<float>& signal, const std::vector<float>& kernel) {
    int signal_length = signal.size();
    int kernel_length = kernel.size();
    int result_length = signal_length + kernel_length - 1;
    std::vector<float> result(result_length, 0);

    for (int i = 0; i < signal_length; ++i) {
        for (int j = 0; j < kernel_length; ++j) {
            float kernel_elem = kernel[j];
            float signal_elem = signal[i];
            int result_idx = i + j;
            result[result_idx] += kernel_elem * signal_elem;
        }
    }

    return result;
}


int main() {
    int signal_length, kernel_length;


    signal_length = 30000;
    kernel_length = 2400;


    std::vector<float> signal(signal_length);
    std::vector<float> kernel(kernel_length);

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10, 10);

    // 生成随机信号
    for (auto& elem : signal) {
        elem = dis(gen);
    }

    // 生成随机卷积核
    for (auto& elem : kernel) {
        elem = dis(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> result = convolution_avx2(signal, kernel);
    auto end = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double> elapsed = end - start;
    std::cout << "计算耗时: " << elapsed.count() << " 秒" << std::endl;
    return 0;
}
