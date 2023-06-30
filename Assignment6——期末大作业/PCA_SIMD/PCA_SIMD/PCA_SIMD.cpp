#include <iostream>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <chrono>
// 计算数据的均值
std::vector<double> computeMean(const std::vector<std::vector<double>>& data) {
    int rows = data.size();
    int cols = data[0].size();

    std::vector<double> mean(cols, 0.0);
    for (int j = 0; j < cols; ++j) {
        double sum = 0.0;
        for (int i = 0; i < rows; ++i) {
            sum += data[i][j];
        }
        mean[j] = sum / rows;
    }
    return mean;
}

// 数据中心化（SIMD优化）
void centerDataSIMD(std::vector<std::vector<double>>& data, const std::vector<double>& mean) {
    int rows = data.size();
    int cols = data[0].size();

    // 使用SIMD指令对数据中心化进行优化
    for (int j = 0; j < cols; ++j) {
        double mean_value = mean[j];

        // 使用SIMD加载均值
        __m256d mean_vec = _mm256_set1_pd(mean_value);

        // 数据中心化
        for (int i = 0; i < rows; i += 4) {
            // 使用SIMD加载数据
            __m256d data_vec = _mm256_loadu_pd(&data[i][j]);

            // 数据减去均值
            __m256d centered_vec = _mm256_sub_pd(data_vec, mean_vec);

            // 存储结果
            _mm256_storeu_pd(&data[i][j], centered_vec);
        }
    }
}

// 计算协方差矩阵（SIMD优化）
std::vector<std::vector<double>> computeCovarianceMatrixSIMD(const std::vector<std::vector<double>>& data) {
    int rows = data.size();
    int cols = data[0].size();

    std::vector<std::vector<double>> covariance(cols, std::vector<double>(cols, 0.0));
    for (int i = 0; i < cols; ++i) {
        for (int j = i; j < cols; ++j) {
            double sum = 0.0;

            // 使用SIMD进行求和
            for (int k = 0; k < rows; k += 4) {
                __m256d data_vec1 = _mm256_loadu_pd(&data[k][i]);
                __m256d data_vec2 = _mm256_loadu_pd(&data[k][j]);

                __m256d mul_result = _mm256_mul_pd(data_vec1, data_vec2);
                __m256d sum_result = _mm256_hadd_pd(mul_result, mul_result);
                __m128d sum_low = _mm256_extractf128_pd(sum_result, 0);
                __m128d sum_high = _mm256_extractf128_pd(sum_result, 1);
                __m128d sum_final = _mm_add_pd(sum_low, sum_high);

                double sum_temp;
                _mm_store_sd(&sum_temp, sum_final);
                sum += sum_temp;
            }

            covariance[i][j] = sum / (rows - 1);
            covariance[j][i] = sum / (rows - 1);
        }
    }
    return covariance;
}

// 特征值分解
void eigenDecomposition(const std::vector<std::vector<double>>& covariance, std::vector<double>& eigenvalues, std::vector<std::vector<double>>& eigenvectors) {
    int cols = covariance.size();

    // 省略特征值分解的具体实现，可以使用第三方库如Eigen进行计算
    // 这里简化为将协方差矩阵的对角线元素作为特征值，单位矩阵作为特征向量
    for (int i = 0; i < cols; ++i) {
        eigenvalues[i] = covariance[i][i];
        for (int j = 0; j < cols; ++j) {
            eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

int main() {
    int rows = 1000;  // 数据行数
    int cols = 100;   // 数据列数
    int k = 10;       // 选择前k个主成分

    // 生成随机数据矩阵
    std::vector<std::vector<double>> data(rows, std::vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
    auto start = std::chrono::high_resolution_clock::now();
    // 计算数据均值
    std::vector<double> mean = computeMean(data);

    // 数据中心化（SIMD优化）
    centerDataSIMD(data, mean);

    // 计算协方差矩阵（SIMD优化）
    std::vector<std::vector<double>> covariance = computeCovarianceMatrixSIMD(data);

    // 特征值分解
    std::vector<double> eigenvalues(cols);
    std::vector<std::vector<double>> eigenvectors(cols, std::vector<double>(cols, 0.0));
    eigenDecomposition(covariance, eigenvalues, eigenvectors);

    // 选择前k个主成分
    std::cout << "Top " << k << " Principal Components:" << std::endl;
    for (int i = 0; i < k; ++i) {
        std::cout << "Eigenvalue " << i + 1 << ": " << eigenvalues[i] << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "训练耗时: " << elapsed.count() << " 秒" << std::endl;
    return 0;
}
