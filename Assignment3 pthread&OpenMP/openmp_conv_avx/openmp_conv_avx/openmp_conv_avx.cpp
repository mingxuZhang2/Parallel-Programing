#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <random>
#include <omp.h>
using namespace std;

int main() {
    int signal_length = 100000;
    int kernel_length = 8000;
    int num_threads = 4;
    double tim[50] = { 0.0 };
    vector<float> signal(signal_length);
    vector<float> kernel(kernel_length);

    // 随机数生成器
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-10, 10);

    // 生成随机信号
    for (auto& elem : signal) {
        elem = dis(gen);
    }

    // 生成随机卷积核
    for (auto& elem : kernel) {
        elem = dis(gen);
    }

    int result_length = signal_length + kernel_length - 1;
    vector<float> result(result_length, 0);

    int chunk_size = signal_length / num_threads;

    auto start = chrono::high_resolution_clock::now();

    // 使用 OpenMP 进行并行化
#pragma omp parallel num_threads(num_threads)
    {
        int thread_num = omp_get_thread_num();
        int start = thread_num * chunk_size;
        int end = (thread_num == num_threads - 1) ? signal_length : (thread_num + 1) * chunk_size;

        __m256 signal_reg, kernel_reg, mul_reg, add_reg;
        int kernel_aligned_length = (kernel_length + 7) / 8 * 8;

        for (int i = start; i < end; ++i) {
            for (int j = 0; j < kernel_aligned_length; j += 8) {
                int result_idx = i + j;
                if (j < kernel_length) {
                    signal_reg = _mm256_set1_ps(signal[i]);
                    kernel_reg = _mm256_loadu_ps(&kernel[j]);
                    mul_reg = _mm256_mul_ps(signal_reg, kernel_reg);
                    add_reg = _mm256_loadu_ps(&result[result_idx]);
                    add_reg = _mm256_add_ps(add_reg, mul_reg);
                    _mm256_storeu_ps(&result[result_idx], add_reg);
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "计算耗时: " << elapsed.count() << " 秒" << endl;
    return 0;
}
