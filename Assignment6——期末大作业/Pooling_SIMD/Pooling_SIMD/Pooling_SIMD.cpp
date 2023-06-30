#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <immintrin.h>

// 池化操作
std::vector<double> performPoolingSIMD(const std::vector<double>& data) {
    int num_samples = data.size();

    // 向下取整至SIMD寄存器的元素个数
    int simd_size = num_samples / 4 * 4;

    // SIMD操作
    std::vector<double> pooled_data(num_samples);

    // 使用SIMD寄存器来存储数据
    __m256d max_values = _mm256_loadu_pd(&data[0]);

    // 每次处理4个元素，以SIMD方式进行最大值计算
    for (int i = 4; i < simd_size; i += 4) {
        __m256d current_values = _mm256_loadu_pd(&data[i]);
        max_values = _mm256_max_pd(max_values, current_values);
    }

    // 将SIMD寄存器中的数据存储回数组
    _mm256_storeu_pd(&pooled_data[0], max_values);

    // 处理剩余的不足4个元素
    for (int i = simd_size; i < num_samples; ++i) {
        if (data[i] > pooled_data[i]) {
            pooled_data[i] = data[i];
        }
    }

    return pooled_data;
}

int main() {
    int num_samples = 1000000;

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // 生成随机数据
    std::vector<double> data(num_samples);
    for (auto& elem : data) {
        elem = dis(gen);
    }

    // SIMD优化的池化操作计时开始
    auto startTime = std::chrono::high_resolution_clock::now();

    // 执行SIMD优化的池化操作
    std::vector<double> pooled_data = performPoolingSIMD(data);

    // SIMD优化的池化操作计时结束
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    std::cout << "Total SIMD pooling time: " << elapsedTime.count() << " seconds" << std::endl;

    return 0;
}
