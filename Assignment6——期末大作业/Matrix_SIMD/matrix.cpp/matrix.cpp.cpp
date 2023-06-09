﻿#include <iostream>
#include <immintrin.h>
#include <vector>

std::vector<std::vector<double>> matrix_multiply_avx2(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();
    std::vector<std::vector<double>> C(m, std::vector<double>(p, 0));
    __m256d A_vec, B_vec, C_vec;
    __m128d C_vec_low, C_vec_high;
    double C_elem[4];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C_vec = _mm256_setzero_pd();
            for (int k = 0; k < n; k += 4) {
                A_vec = _mm256_load_pd(&A[i][k]);
                B_vec = _mm256_load_pd(&B[k][j]);
                C_vec = _mm256_add_pd(C_vec, _mm256_mul_pd(A_vec, B_vec));
            }
            C_vec_low = _mm256_extractf128_pd(C_vec, 0);
            C_vec_high = _mm256_extractf128_pd(C_vec, 1);
            _mm_store_sd(&C_elem[0], C_vec_low);
            _mm_store_sd(&C_elem[1], _mm_unpackhi_pd(C_vec_low, C_vec_low));
            _mm_store_sd(&C_elem[2], C_vec_high);
            _mm_store_sd(&C_elem[3], _mm_unpackhi_pd(C_vec_high, C_vec_high));
            C[i][j] = C_elem[0] + C_elem[1] + C_elem[2] + C_elem[3];
        }
    }
    return C;
}

int main() {
    int m, n, p;
    std::cout << "Enter the number of rows of matrix A: ";
    std::cin >> m;
    std::cout << "Enter the number of columns of matrix A: ";
    std::cin >> n;
    std::cout << "Enter the number of columns of matrix B: ";
    std::cin >> p;

    std::vector<std::vector<double>> A(m, std::vector<double>(n));
    std::vector<std::vector<double>> B(n, std::vector<double>(p));

    std::cout << "Enter the elements of matrix A: " << std::endl;
    for (auto& row : A) {
        for (auto& elem : row) {
            std::cin >> elem;
        }
    }

    std::cout << "Enter the elements of matrix B: " << std::endl;
    for (auto& row : B) {
        for (auto& elem : row) {
            std::cin >> elem;
        }
    }

    std::vector<std::vector<double>> C = matrix_multiply_avx2(A, B);

    std::cout << "The result matrix is: " << std::endl;
    for (const auto& row : C) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
