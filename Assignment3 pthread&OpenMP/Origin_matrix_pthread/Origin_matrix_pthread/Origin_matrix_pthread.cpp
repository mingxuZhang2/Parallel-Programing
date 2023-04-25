#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <pthread.h>
#pragma comment(lib, "pthreadVC2.lib")
struct MultiplyArgs {
    const std::vector<std::vector<double>>* A;
    const std::vector<std::vector<double>>* B;
    std::vector<std::vector<double>>* C;
    int start_row;
    int end_row;
};

void* multiply_thread(void* args) {
    MultiplyArgs* multiplyArgs = static_cast<MultiplyArgs*>(args);
    const auto& A = *multiplyArgs->A;
    const auto& B = *multiplyArgs->B;
    auto& C = *multiplyArgs->C;

    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();

    for (int i = multiplyArgs->start_row; i < multiplyArgs->end_row; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return nullptr;
}

int main() {
    int m = 500;
    int n = 500;
    int p = 500;
    int num_threads = 4;

    std::vector<std::vector<double>> A(m, std::vector<double>(n));
    std::vector<std::vector<double>> B(n, std::vector<double>(p));
    std::vector<std::vector<double>> C(m, std::vector<double>(p, 0.0));

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

    auto start = std::chrono::steady_clock::now();

    std::vector<pthread_t> threads(num_threads);
    std::vector<MultiplyArgs> thread_args(num_threads);
    int rows_per_thread = m / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_args[i] = { &A, &B, &C, i * rows_per_thread, (i + 1) * rows_per_thread };
        if (i == num_threads - 1) {
            thread_args[i].end_row = m;
        }
        pthread_create(&threads[i], nullptr, multiply_thread, &thread_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "矩阵乘法完成" << std::endl;
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "计算耗时: " << elapsed.count() << " 秒" << std::endl;
    return 0;
}