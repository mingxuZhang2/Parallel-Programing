#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>

struct ThreadData {
    int start;
    int end;
    const std::vector<std::vector<double>>& data;
    std::vector<double>& mean;
};

// 计算数据的均值（线程函数）
void* computeMeanThread(void* arg) {
    ThreadData* threadData = static_cast<ThreadData*>(arg);
    const std::vector<std::vector<double>>& data = threadData->data;
    std::vector<double>& mean = threadData->mean;

    int rows = data.size();
    int cols = data[0].size();

    for (int j = threadData->start; j < threadData->end; ++j) {
        double sum = 0.0;
        for (int i = 0; i < rows; ++i) {
            sum += data[i][j];
        }
        mean[j] = sum / rows;
    }

    pthread_exit(nullptr);
}

// 计算数据的均值
std::vector<double> computeMean(const std::vector<std::vector<double>>& data) {
    int cols = data[0].size();
    std::vector<double> mean(cols, 0.0);

    int numThreads = 4;  // 线程数
    std::vector<pthread_t> threads(numThreads);
    std::vector<ThreadData> threadData(numThreads);

    // 创建线程
    int step = cols / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * step;
        int end = (i == numThreads - 1) ? cols : (i + 1) * step;

        threadData[i] = { start, end, data, mean };
        pthread_create(&threads[i], nullptr, computeMeanThread, &threadData[i]);
    }

    // 等待线程完成
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    return mean;
}

int main() {
    int rows = 1000;  // 数据行数
    int cols = 100;   // 数据列数

    // 生成随机数据矩阵
    std::vector<std::vector<double>> data(rows, std::vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }

    // 计算数据均值
    std::vector<double> mean = computeMean(data);

    // 输出结果
    std::cout << "Mean:" << std::endl;
    for (int j = 0; j < cols; ++j) {
        std::cout << mean[j] << " ";
    }
    std::cout << std::endl;

    return 0;
}
