#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <random>
void matrix_multiply_avx2(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; j += 8) {
            __m256 c_line = _mm256_setzero_ps();
            for (int k = 0; k < n; ++k) {
                __m256 a_elem = _mm256_set1_ps(a[i * n + k]);
                __m256 b_line = _mm256_loadu_ps(&b[k * p + j]);
                c_line = _mm256_add_ps(c_line, _mm256_mul_ps(a_elem, b_line));
            }
            _mm256_storeu_ps(&c[i * p + j], c_line);
        }
    }
}
int main() {
    int m = 500;
    int n = 500;
    int p = 500;

    std::vector<float> a(m * n);
    std::vector<float> b(n * p);
    std::vector<float> c(m * p);

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10, 10);

    // 生成随机矩阵 A
    for (int i = 0; i < m * n; ++i) {
        a[i] = dis(gen);
    }

    // 生成随机矩阵 B
    for (int i = 0; i < n * p; ++i) {
        b[i] = dis(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiply_avx2(a, b, c, m, n, p);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "矩阵乘法完成" << std::endl;
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "计算耗时: " << elapsed.count() << " 秒" << std::endl;

    return 0;
}
/*

10 10 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
*/