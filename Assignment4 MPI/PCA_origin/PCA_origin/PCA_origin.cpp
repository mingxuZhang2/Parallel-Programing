#include <iostream>
#include <vector>
#include <cmath>

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

// 数据中心化
void centerData(std::vector<std::vector<double>>& data, const std::vector<double>& mean) {
    int rows = data.size();
    int cols = data[0].size();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] -= mean[j];
        }
    }
}

// 计算协方差矩阵
std::vector<std::vector<double>> computeCovarianceMatrix(const std::vector<std::vector<double>>& data) {
    int rows = data.size();
    int cols = data[0].size();

    std::vector<std::vector<double>> covariance(cols, std::vector<double>(cols, 0.0));
    for (int i = 0; i < cols; ++i) {
        for (int j = i; j < cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < rows; ++k) {
                sum += data[k][i] * data[k][j];
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
    // 计算数据均值
    std::vector<double> mean = computeMean(data);

    // 数据中心化
    centerData(data, mean);

    // 计算协方差矩阵
    std::vector<std::vector<double>> covariance = computeCovarianceMatrix(data);

    // 特征值分解
    std::vector<double> eigenvalues(cols);
    std::vector<std::vector<double>> eigenvectors(cols, std::vector<double>(cols, 0.0));
    eigenDecomposition(covariance, eigenvalues, eigenvectors);

    // 选择前k个主成分
    std::cout << "Top " << k << " Principal Components:" << std::endl;
    for (int i = 0; i < k; ++i) {
        std::cout << "Eigenvalue " << i + 1 << ": " << eigenvalues[i] << std::endl;
    }

    return 0;
}
