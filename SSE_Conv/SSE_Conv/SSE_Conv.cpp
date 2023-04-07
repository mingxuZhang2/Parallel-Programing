#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <immintrin.h>
#include <chrono>
#include <random>

std::vector<float> convolution_sse(const std::vector<float>& signal, const std::vector<float>& kernel) {
    int signal_length = signal.size();
    int kernel_length = kernel.size();
    int result_length = signal_length + kernel_length - 1;
    std::vector<float> result(result_length, 0);
    for (int i = 0; i < signal_length; ++i) {
        for (int j = 0; j < kernel_length; ++j) {
            int result_idx = i + j;

            if (i % 4 == 0 && i + 3 < signal_length) {
                __m128 kernel_elem = _mm_set1_ps(kernel[j]);
                __m128 signal_elems = _mm_loadu_ps(&signal[i]);
                __m128 result_elems = _mm_loadu_ps(&result[result_idx]);
                result_elems = _mm_add_ps(result_elems, _mm_mul_ps(signal_elems, kernel_elem));
                _mm_storeu_ps(&result[result_idx], result_elems);
                i += 3;
            }
            else {
                result[result_idx] += signal[i] * kernel[j];
            }
        }
    }
    return result;
}
int main() {
    int signal_length = 2000;
    int kernel_length = 160;

    std::vector<float> signal(signal_length);
    std::vector<float> kernel(kernel_length);

    for (auto& elem : signal) {
        elem = rand() % 10;
    }

    for (auto& elem : kernel) {
        elem = rand() % 10;
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> result = convolution_sse(signal, kernel);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "卷积操作完成" << std::endl;
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "计算耗时: " << elapsed.count() << " 秒" << std::endl;
    

    return 0;
}
