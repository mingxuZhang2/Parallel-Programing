#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>

// 生成随机数据矩阵
void generateRandomData(std::vector<double>& data, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) {
        data[i] = dis(gen);
    }
}

// 数据中心化
void centerData(std::vector<double>& data, int rows, int cols) {
    std::vector<double> colMeans(cols, 0.0);

    // 计算每列的均值
    for (int j = 0; j < cols; ++j) {
        double sum = 0.0;
        for (int i = 0; i < rows; ++i) {
            sum += data[i * cols + j];
        }
        colMeans[j] = sum / rows;
    }

    // 中心化
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i * cols + j] -= colMeans[j];
        }
    }
}

// 计算局部协方差矩阵
void computeLocalCovariance(const std::vector<double>& localData, int localRows, int cols, std::vector<double>& localCovariance) {
    for (int i = 0; i < cols; ++i) {
        for (int j = i; j < cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < localRows; ++k) {
                sum += localData[k * cols + i] * localData[k * cols + j];
            }
            localCovariance[i * cols + j] = sum;
            localCovariance[j * cols + i] = sum;
        }
    }
}

// 归约全局协方差矩阵
void reduceGlobalCovariance(std::vector<double>& localCovariance, int cols) {
    std::vector<double> globalCovariance(cols * cols, 0.0);
    MPI_Allreduce(localCovariance.data(), globalCovariance.data(), cols * cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    localCovariance = globalCovariance;
}

// 特征值分解
void eigenDecomposition(const std::vector<double>& covariance, int cols, std::vector<double>& eigenvalues, std::vector<double>& eigenvectors) {
    // 省略特征值分解的具体实现，可以使用第三方库如Eigen进行计算
    // 这里简化为将协方差矩阵的对角线元素作为特征值，单位矩阵作为特征向量
    for (int i = 0; i < cols; ++i) {
        eigenvalues[i] = covariance[i * cols + i];
        for (int j = 0; j < cols; ++j) {
            eigenvectors[i * cols + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rows = 1000;  // 数据行数
    int cols = 100;   // 数据列数
    int k = 10;       // 选择前k个主成分

    int localRows = rows / worldSize;
    std::vector<double> localData(localRows * cols);

    if (rank == 0) {
        std::vector<double> data(rows * cols);
        generateRandomData(data, rows, cols);

        // 将数据分发给各个进程
        MPI_Scatter(data.data(), localRows * cols, MPI_DOUBLE, localData.data(), localRows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Scatter(nullptr, localRows * cols, MPI_DOUBLE, localData.data(), localRows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // 数据中心化
    centerData(localData, localRows, cols);

    // 计算局部协方差矩阵
    std::vector<double> localCovariance(cols * cols);
    computeLocalCovariance(localData, localRows, cols, localCovariance);

    // 归约全局协方差矩阵
    reduceGlobalCovariance(localCovariance, cols);

    if (rank == 0) {
        std::vector<double> eigenvalues(cols);
        std::vector<double> eigenvectors(cols * cols);

        // 特征值分解
        eigenDecomposition(localCovariance, cols, eigenvalues, eigenvectors);

        // 选择前k个主成分
        std::cout << "Top " << k << " Principal Components:" << std::endl;
        for (int i = 0; i < k; ++i) {
            std::cout << "Eigenvalue " << i + 1 << ": " << eigenvalues[i] << std::endl;
        }
    }

    MPI_Finalize();

    return 0;

}
