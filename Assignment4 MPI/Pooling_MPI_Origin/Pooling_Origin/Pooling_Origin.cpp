#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mpi.h>

// 池化操作
std::vector<double> performPooling(const std::vector<double>& data, int rank, int worldSize) {
    int num_samples = data.size();

    // 计算每个进程负责的数据范围
    int chunk_size = num_samples / worldSize;
    int start = rank * chunk_size;
    int end = (rank == worldSize - 1) ? num_samples : (rank + 1) * chunk_size;

    // 池化操作
    std::vector<double> pooled_data(chunk_size);
    for (int i = start; i < end; ++i) {
        // 池化操作，例如取平均值或最大值
        // 这里示例为取最大值
        double max_value = data[i];
        for (int j = i + 1; j < num_samples; ++j) {
            if (data[j] > max_value) {
                max_value = data[j];
            }
        }
        pooled_data[i - start] = max_value;
    }

    // 进行结果的全局归约
    std::vector<double> global_pooled_data(num_samples);
    MPI_Allgather(pooled_data.data(), chunk_size, MPI_DOUBLE,
        global_pooled_data.data(), chunk_size, MPI_DOUBLE,
        MPI_COMM_WORLD);

    return global_pooled_data;
}

int main(int argc, char** argv) {
    // MPI初始化
    MPI_Init(&argc, &argv);

    // 获取MPI进程总数和当前进程编号
    int worldSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    // 池化操作计时开始
    auto startTime = std::chrono::high_resolution_clock::now();

    // 执行池化操作
    std::vector<double> pooled_data = performPooling(data, rank, worldSize);

    // 池化操作计时结束
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    // 计算池化操作的总时间
    double totalPoolingTime;
    MPI_Reduce(&elapsedTime, &totalPoolingTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Total pooling time: " << totalPoolingTime.count() << " seconds" << std::endl;
    }

    // MPI结束
    MPI_Finalize();

    return 0;
}
