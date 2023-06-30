#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <pthread.h>

// 线程参数结构体
struct ThreadArgs {
    const std::vector<double>* data;
    std::vector<double>* pooled_data;
    int start_index;
    int end_index;

    // 手动编写赋值操作
    ThreadArgs& operator=(const ThreadArgs& other) {
        data = other.data;
        pooled_data = other.pooled_data;
        start_index = other.start_index;
        end_index = other.end_index;
        return *this;
    }
};

// 线程执行函数
void* performPoolingPThread(void* arg) {
    ThreadArgs* thread_args = static_cast<ThreadArgs*>(arg);
    const std::vector<double>& data = *(thread_args->data);
    std::vector<double>& pooled_data = *(thread_args->pooled_data);
    int start_index = thread_args->start_index;
    int end_index = thread_args->end_index;

    // 执行池化操作，例如取最大值
    for (int i = start_index; i < end_index; ++i) {
        double max_value = data[i];
        for (int j = i + 1; j < data.size(); ++j) {
            if (data[j] > max_value) {
                max_value = data[j];
            }
        }
        pooled_data[i] = max_value;
    }

    return nullptr;
}

// 池化操作
std::vector<double> performPoolingPThread(const std::vector<double>& data, int num_threads) {
    int num_samples = data.size();

    // 池化操作结果
    std::vector<double> pooled_data(num_samples);

    // 创建线程参数和线程对象
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadArgs> thread_args(num_threads);

    // 计算每个线程需要处理的数据范围
    int chunk_size = num_samples / num_threads;
    int start_index = 0;
    int end_index = chunk_size;

    // 创建线程并执行池化操作
    for (int i = 0; i < num_threads; ++i) {
        thread_args[i].data = &data;
        thread_args[i].pooled_data = &pooled_data;
        thread_args[i].start_index = start_index;
        thread_args[i].end_index = end_index;
        pthread_create(&threads[i], nullptr, performPoolingPThread, &thread_args[i]);
        start_index = end_index;
        end_index = (i == num_threads - 2) ? num_samples : end_index + chunk_size;
    }

    // 等待所有线程执行完毕
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    return pooled_data;
}

int main() {
    int num_samples = 1000000;
    int num_threads = 4;

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // 生成随机数据
    std::vector<double> data(num_samples);
    for (auto& elem : data) {
        elem = dis(gen);
    }

    // Pthread优化的池化操作计时开始
    auto startTime = std::chrono::high_resolution_clock::now();

    // 执行Pthread优化的池化操作
    std::vector<double> pooled_data = performPoolingPThread(data, num_threads);

    // Pthread优化的池化操作计时结束
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    std::cout << "Total Pthread pooling time: " << elapsedTime.count() << " seconds" << std::endl;

    return 0;
}
