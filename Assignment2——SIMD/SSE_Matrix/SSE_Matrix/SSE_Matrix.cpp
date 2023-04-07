#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <immintrin.h>
#include <chrono>
#include <random>
std::vector<std::vector<float>> matrix_multiply_sse(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();
    std::vector<std::vector<float>> C(m, std::vector<float>(p, 0.0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            __m128 sum = _mm_setzero_ps();
            for (int k = 0; k < n; k += 4) {
                __m128 a = _mm_loadu_ps(&A[i][k]);
                __m128 b = _mm_loadu_ps(&B[k][j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }
            float temp[4];
            _mm_storeu_ps(temp, sum);
            C[i][j] = temp[0] + temp[1] + temp[2] + temp[3];

            for (int k = n - n % 4; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

int main() {
    srand(time(0));

    int m = 500;
    int n = 500;
    int p = 500;

    std::vector<std::vector<float>> A(m, std::vector<float>(n));
    std::vector<std::vector<float>> B(n, std::vector<float>(p));

    for (auto& row : A) {
        for (auto& elem : row) {
            elem = rand() % 10;
        }
    }

    for (auto& row : B) {
        for (auto& elem : row) {
            elem = rand() % 10;
        }
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> C = matrix_multiply_sse(A, B);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "乘法操作完成" << std::endl;
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "计算耗时: " << elapsed.count() << " 秒" << std::endl;
    return 0;
}
