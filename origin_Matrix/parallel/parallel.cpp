#include <iostream>
#include <vector>
#include <chrono>
#include <random>

std::vector<std::vector<double>> matrix_multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();
    std::vector<std::vector<double>> C(m, std::vector<double>(p, 0.0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

int main() {
    int m = 500;
    int n = 500;
    int p = 500;

    std::vector<std::vector<double>> A(m, std::vector<double>(n));
    std::vector<std::vector<double>> B(n, std::vector<double>(p));

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10, 10);

    // 生成随机矩阵 A
    for (auto& row : A) {
        for (auto& elem : row) {
            elem = dis(gen);
        }
    }
    // 生成随机矩阵 B
    for (auto& row : B) {
        for (auto& elem : row) {
            elem = dis(gen);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> C = matrix_multiply(A, B);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "矩阵乘法完成" << std::endl;
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "计算耗时: " << elapsed.count() << " 秒" << std::endl;

    return 0;

}
