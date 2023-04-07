#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <random>
using namespace std;
#include <immintrin.h>
#include <vector>

std::vector<float> convolution_avx2(const std::vector<float>& signal, const std::vector<float>& kernel) {
    int signal_length = signal.size();
    int kernel_length = kernel.size();
    int result_length = signal_length + kernel_length - 1;
    std::vector<float> result(result_length, 0);

    for (int i = 0; i < signal_length; ++i) {
        for (int j = 0; j < kernel_length; ++j) {
            int result_idx = i + j;

            if (i % 8 == 0 && i + 7 < signal_length) {
                // AVX2 optimization when 8 elements can be processed at once
                __m256 kernel_elem = _mm256_set1_ps(kernel[j]);
                __m256 signal_elems = _mm256_loadu_ps(&signal[i]);
                __m256 result_elems = _mm256_loadu_ps(&result[result_idx]);
                result_elems = _mm256_add_ps(result_elems, _mm256_mul_ps(kernel_elem, signal_elems));
                _mm256_storeu_ps(&result[result_idx], result_elems);
                i += 7;
            }
            else {
                // Handling the boundary case with regular loop
                result[result_idx] += kernel[j] * signal[i];
            }
        }
    }

    return result;
}

int main() {
    int signal_length, kernel_length;


    signal_length=1600;
    kernel_length=80;
    

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
    system("pause");
    return 0;
}
