#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// 池化操作
std::vector<double> performPooling(const std::vector<double>& data) {
    int num_samples = data.size();

    // 池化操作
    std::vector<double> pooled_data(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        // 池化操作，例如取平均值或最大值
        // 这里示例为取最大值
        double max_value = data[i];
        for (int j = i + 1; j < num_samples; ++j) {
            if (data[j] > max_value) {
                max_value = data[j];
            }
        }
        pooled_data[i] = max_value;
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

    // 池化操作计时开始
    auto startTime = std::chrono::high_resolution_clock::now();

    // 执行池化操作
    std::vector<double> pooled_data = performPooling(data);

    // 池化操作计时结束
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    std::cout << "Total pooling time: " << elapsedTime.count() << " seconds" << std::endl;

    return 0;
}
