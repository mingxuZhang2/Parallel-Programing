
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include <immintrin.h>

void performPCA(std::vector<std::vector<double>>& data, std::vector<std::vector<double>>& eigen_vectors, int rank, int worldSize, int num_threads) {
    int num_samples = data.size();
    int num_features = data[0].size();
    int num_eigen_vectors = eigen_vectors.size();

    // 计算每个进程负责的工作范围
    int chunk_size = num_samples / (num_threads * worldSize);
    int start = rank * num_threads * chunk_size;
    int end = (rank == worldSize - 1) ? num_samples : (rank + 1) * num_threads * chunk_size;

    // 计算数据的均值
    std::vector<double> mean(num_features, 0.0);
#pragma omp parallel for num_threads(num_threads) reduction(+:mean[:num_features])
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            mean[j] += data[i][j];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, mean.data(), num_features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (auto& elem : mean) {
        elem /= num_samples * worldSize;
    }

    // 将数据减去均值
#pragma omp parallel for num_threads(num_threads)
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < num_features; ++j) {
            data[i][j] -= mean[j];
        }
    }

    // 计算协方差矩阵的一部分
    std::vector<std::vector<double>> covariance(num_features, std::vector<double>(num_features, 0.0));
#pragma omp parallel for num_threads(num_threads)
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < num_features; ++j) {
            for (int k = 0; k < num_features; ++k) {
                covariance[j][k] += data[i][j] * data[i][k];
            }
        }
    }

    // 进行结果的全局归约
    for (int i = 0; i < num_features; ++i) {
        MPI_Allreduce(MPI_IN_PLACE, covariance[i].data(), num_features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // 计算特征值和特征向量的一部分
    std::vector<std::vector<double>> partial_eigen_vectors(num_eigen_vectors, std::vector<double>(num_features, 0.0));
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_eigen_vectors; ++i) {
        for (int j = 0; j < num_features; j += 4) {
            __m256d sum = _mm256_setzero_pd();
            for (int k = 0; k < num_features; ++k) {
                __m256d data_vec = _mm256_set_pd(data[k][j + 3], data[k][j + 2], data[k][j + 1], data[k][j]);
                __m256d eigen_vec = _mm256_set_pd(eigen_vectors[i][j + 3], eigen_vectors[i][j + 2], eigen_vectors[i][j + 1], eigen_vectors[i][j]);
                __m256d product = _mm256_mul_pd(data_vec, eigen_vec);
                sum = _mm256_add_pd(sum, product);
            }
            _mm256_storeu_pd(&partial_eigen_vectors[i][j], sum);
        }
    }

    // 进行结果的全局归约
    for (int i = 0; i < num_eigen_vectors; ++i) {
        MPI_Allreduce(MPI_IN_PLACE, partial_eigen_vectors[i].data(), num_features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // 更新部分特征向量的一部分
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_eigen_vectors; ++i) {
        for (int j = 0; j < num_features; ++j) {
            eigen_vectors[i][j] = partial_eigen_vectors[i][j];
        }
    }
}

int main(int argc, char** argv) {
    // MPI初始化
    MPI_Init(&argc, &argv);

    // 获取MPI进程总数和当前进程编号
    int worldSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_samples = 100000;
    int num_features = 1000;
    int num_eigen_vectors = 10;
    int num_threads = 4;

    // 数据生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);

    // 生成随机数据
    std::vector<std::vector<double>> data(num_samples, std::vector<double>(num_features));
    for (auto& sample : data) {
        for (auto& elem : sample) {
            elem = dis(gen);
        }
    }

    // 生成随机特征向量
    std::vector<std::vector<double>> eigen_vectors(num_eigen_vectors, std::vector<double>(num_features));
    for (auto& eigen_vector : eigen_vectors) {
        for (auto& elem : eigen_vector) {
            elem = dis(gen);
        }
    }

    // PCA操作计时开始
    auto startTime = std::chrono::high_resolution_clock::now();

    // 执行PCA操作
    performPCA(data, eigen_vectors, rank, worldSize, num_threads);

    // PCA操作计时结束
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    // 计算PCA操作的总时间
    double totalPCATime;
    MPI_Reduce(&elapsedTime, &totalPCATime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Total PCA time: " << totalPCATime.count() << " seconds" << std::endl;
    }

    // MPI结束
    MPI_Finalize();

    return 0;
}