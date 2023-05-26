#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <random>
#include <mpi.h>

void performEmbedding(const std::vector<int>& data, const std::vector<std::vector<float>>& embeddings, std::vector<std::vector<float>>& result, int rank, int worldSize) {
    int num_data = data.size();
    int embedding_size = embeddings[0].size();
    int result_size = result[0].size();

    // 计算每个进程负责的工作范围
    int chunk_size = num_data / worldSize;
    int start = rank * chunk_size;
    int end = (rank == worldSize - 1) ? num_data : (rank + 1) * chunk_size;

    // 进行Embedding计算
    __m256i indices, embedding_indices;
    __m256 embedding_value, result_value;
    for (int i = start; i < end; ++i) {
        indices = _mm256_set1_epi32(data[i]);

        for (int j = 0; j < embedding_size; j += 8) {
            embedding_indices = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&embeddings[j][0]));
            embedding_value = _mm256_i32gather_ps(reinterpret_cast<const float*>(&embedding_indices), indices, sizeof(float));
            result_value = _mm256_loadu_ps(&result[i][j]);
            result_value = _mm256_add_ps(result_value, embedding_value);
            _mm256_storeu_ps(&result[i][j], result_value);
        }
    }

    // 进行结果的全局归约
    for (int i = 0; i < num_data; ++i) {
        MPI_Allreduce(MPI_IN_PLACE, result[i].data(), result_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    // MPI初始化
    MPI_Init(&argc, &argv);

    // 获取MPI进程总数和当前进程编号
    int worldSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_data = 100000;
    int embedding_size = 200;
    int result_size = embedding_size;

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, embedding_size - 1);

    // 生成随机数据
    std::vector<int> data(num_data);
    for (auto& elem : data) {
        elem = dis(gen);
    }

    // 生成随机嵌入
    std::vector<std::vector<float>> embeddings(embedding_size, std::vector<float>(embedding_size));
    for (auto& row : embeddings) {
        for (auto& elem : row) {
            elem = static_cast<float>(dis(gen));
        }
    }

    // 计算每个进程负责的嵌入向量范围
    int chunk_size = num_data / worldSize;
    int start = rank * chunk_size;
    int end = (rank == worldSize - 1) ? num_data : (rank + 1) * chunk_size;

    // 初始化结果矩阵
    std::vector<std::vector<float>> result(end - start, std::vector<float>(result_size, 0.0));
    // 嵌入操作计时开始
    auto startTime = std::chrono::high_resolution_clock::now();

    // 执行嵌入操作
    performEmbedding(data, embeddings, result, rank, worldSize);

    // 嵌入操作计时结束
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    // 计算嵌入操作的总时间
    double totalEmbeddingTime;
    MPI_Reduce(&elapsedTime, &totalEmbeddingTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Total embedding time: " << totalEmbeddingTime.count() << " seconds" << std::endl;
    }

    // MPI结束
    MPI_Finalize();

    return 0;
}
