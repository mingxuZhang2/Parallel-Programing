#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <immintrin.h>
#include <mpi.h>

// 池化操作
std::vector<double> performPooling(const std::vector<double>& data, int rank, int worldSize) {
    int num_samples = data.size();
    int chunk_size = num_samples / worldSize;
    int start = rank * chunk_size;
    int end = (rank == worldSize - 1) ? num_samples : (rank + 1) * chunk_size;

    // 池化操作
    std::vector<double> pooled_data(end - start);
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

    return pooled_data;
}

int main(int argc, char** argv) {
    // MPI初始化
    MPI_Init(&argc, &argv);

    // 获取MPI进程总数和当前进程编号
    int worldSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_samples = 1000000;

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // 主进程生成随机数据
    std::vector<double> data(num_samples);
    if (rank == 0) {
        for (auto& elem : data) {
            elem = dis(gen);
        }
    }

    // 广播数据给其他进程
    MPI_Bcast(data.data(), num_samples, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 数据分发给各进程
    int chunk_size = num_samples / worldSize;
    int start = rank * chunk_size;
    int end = (rank == worldSize - 1) ? num_samples : (rank + 1) * chunk_size;
    std::vector<double> local_data(end - start);
    MPI_Scatter(data.data(), chunk_size, MPI_DOUBLE, local_data.data(), chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 池化操作计时开始
    auto startTime = std::chrono::high_resolution_clock::now();

    // 执行池化操作
    std::vector<double> pooled_data = performPooling(local_data, rank, worldSize);

    // 汇总池化结果到主进程
    std::vector<double> gathered_data;
    if (rank == 0) {
        gathered_data.resize(num_samples);
    }
    MPI_Gather(pooled_data.data(), pooled_data.size(), MPI_DOUBLE, gathered_data.data(), pooled_data.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 池化操作计时结束
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    if (rank == 0) {
        std::cout << "Total pooling time: " << elapsedTime.count() << " seconds" << std::endl;
    }

    // MPI结束
    MPI_Finalize();

    return 0;

}
