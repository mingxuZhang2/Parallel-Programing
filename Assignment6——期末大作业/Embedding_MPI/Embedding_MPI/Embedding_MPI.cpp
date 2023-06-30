#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mpi.h>

void performEmbedding(const std::vector<int>& input, const std::vector<std::vector<float>>& embeddings, std::vector<std::vector<float>>& output, int rank, int worldSize) {
    int input_size = input.size();
    int embedding_dim = embeddings[0].size();

    // 计算每个进程负责的工作范围
    int chunk_size = input_size / worldSize;
    int start = rank * chunk_size;
    int end = (rank == worldSize - 1) ? input_size : (rank + 1) * chunk_size;

    // 进行嵌入计算
    for (int i = start; i < end; ++i) {
        int index = input[i];
        for (int j = 0; j < embedding_dim; ++j) {
            output[i][j] = embeddings[index][j];
        }
    }

    // 进行结果的全局归约
    MPI_Allgather(MPI_IN_PLACE, chunk_size * embedding_dim, MPI_FLOAT,output[start].data(), chunk_size * embedding_dim, MPI_FLOAT,MPI_COMM_WORLD);

}

int main(int argc, char** argv) {
    // MPI初始化
    MPI_Init(&argc, &argv);

    // 获取MPI进程总数和当前进程编号
    int worldSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int input_size = 100000;  // 输入样本的大小
    int embedding_dim = 300;  // 嵌入维度

    // 每个进程的数据
    std::vector<int> input(input_size);  // 输入样本
    std::vector<std::vector<float>> embeddings(input_size, std::vector<float>(embedding_dim));  // 嵌入表
    std::vector<std::vector<float>> output(input_size, std::vector<float>(embedding_dim));  // 输出结果

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> input_dis(0, input_size - 1);
    std::uniform_real_distribution<> embedding_dis(-1.0, 1.0);

    // 主进程生成随机输入样本和嵌入表
    if (rank == 0) {
        // 生成随机输入样本
        for (auto& elem : input) {
            elem = input_dis(gen);
        }

        // 生成随机嵌入表
        for (auto& row : embeddings) {
            for (auto& elem : row) {
                elem = embedding_dis(gen);
            }
        }
    }

    // 将输入样本广播给其他进程
    MPI_Bcast(input.data(), input_size, MPI_INT, 0, MPI_COMM_WORLD);
    // 将嵌入表广播给其他进程
    for (auto& row : embeddings) {
        MPI_Bcast(row.data(), embedding_dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // 嵌入操作计时开始
    auto startTime = std::chrono::high_resolution_clock::now();

    // 执行嵌入操作
    performEmbedding(input, embeddings, output, rank, worldSize);

    // 嵌入操作计时结束
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    // 计算嵌入操作的总时间
    double totalEmbeddingTime;
    MPI_Reduce(&elapsedTime, &totalEmbeddingTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Total embedding time: " << elapsedTime.count() << " seconds" << std::endl;
    }

    // MPI结束
    MPI_Finalize();

    return 0;
}
