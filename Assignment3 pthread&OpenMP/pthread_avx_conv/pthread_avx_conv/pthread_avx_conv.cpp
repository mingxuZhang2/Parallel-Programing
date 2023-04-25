#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <random>
#include <pthread.h>
#pragma comment(lib, "pthreadVC2.lib")
using namespace std;

struct ConvolutionData {
    const vector<float>* signal;
    const vector<float>* kernel;
    vector<float>* result;
    int start;
    int end;
    double elapsed_time; // 新增：存储每个线程的用时
};

void* convolution_thread(void* arg) {
    auto thread_start = std::chrono::high_resolution_clock::now();
    ConvolutionData* data = static_cast<ConvolutionData*>(arg);
    const vector<float>& signal = *data->signal;
    const vector<float>& kernel = *data->kernel;
    vector<float>& result = *data->result;
    int signal_length = signal.size();
    int kernel_length = kernel.size();
    int kernel_aligned_length = (kernel_length + 7) / 8 * 8;

    __m256 signal_reg, kernel_reg, mul_reg, add_reg;

    for (int i = data->start; i < data->end; ++i) {
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
    auto thread_end = std::chrono::high_resolution_clock::now(); // 新增：线程结束计时

    data->elapsed_time = std::chrono::duration<double>(thread_end - thread_start).count(); // 新增：计算并存储线程用时
    return nullptr;
}


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

    // 创建线程和线程数据
    vector<pthread_t> threads(num_threads);
    vector<ConvolutionData> thread_data(num_threads);

    int chunk_size = signal_length / num_threads;

    auto start = chrono::high_resolution_clock::now();

    // 创建并运行线程
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].signal = &signal;
        thread_data[i].kernel = &kernel;
        thread_data[i].result = &result;
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? signal_length : (i + 1) * chunk_size;

        pthread_create(&threads[i], nullptr, convolution_thread, &thread_data[i]);
    }

    // 等待所有线程完成
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
        tim[i] += thread_data[i].elapsed_time;
    }
    for (int i = 0; i < num_threads; i++) {
        std::cout << "线程 " << i << "用时" << tim[i] << endl;

    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "计算耗时: " << elapsed.count() << " 秒" << endl;

    system("pause");
    return 0;
}
