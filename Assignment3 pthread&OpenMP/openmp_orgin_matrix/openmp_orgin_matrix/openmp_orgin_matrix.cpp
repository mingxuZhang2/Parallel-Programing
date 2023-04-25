#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
using namespace std;

int main() {
    int m = 500;
    int n = 500;
    int p = 500;
    int num_threads = 4;

    vector<vector<double>> A(m, vector<double>(n));
    vector<vector<double>> B(n, vector<double>(p));
    vector<vector<double>> C(m, vector<double>(p, 0.0));

    // 随机数生成器
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-10, 10);

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

    auto start = chrono::steady_clock::now();

    // 使用OpenMP并行计算矩阵乘法
    omp_set_num_threads(num_threads);
#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    auto end = chrono::steady_clock::now();
    cout << "矩阵乘法完成" << endl;
    chrono::duration<double> elapsed = end - start;
    cout << "计算耗时: " << elapsed.count() << " 秒" << endl;

    return 0;
}
