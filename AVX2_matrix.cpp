#include <iostream>
#include <vector>
#include <immintrin.h>
using namespace std;
std::vector<std::vector<double>> matrix_multiply_avx2(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();
    std::vector<std::vector<double>> C(m, std::vector<double>(p, 0));
    __m256d A_vec, B_vec, C_vec;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C_vec = _mm256_setzero_pd();
            for (int k = 0; k < n; k += 4) {
                A_vec = _mm256_loadu_pd(&A[i][k]);
                B_vec = _mm256_loadu_pd(&B[k][j]);
                C_vec = _mm256_add_pd(C_vec, _mm256_mul_pd(A_vec, B_vec));
            }
            C[i][j] = C_vec[0] + C_vec[1] + C_vec[2] + C_vec[3];
        }
    }
    return C;
}

int main() {
    freopen("1.in", "r", stdin);
    int m, n, p;
    std::cin >> m;
    std::cin >> n;
    std::cin >> p;

    std::vector<std::vector<double>> A(m, std::vector<double>(n));
    std::vector<std::vector<double>> B(n, std::vector<double>(p));

    for (auto& row : A) {
        for (auto& elem : row) {
            std::cin >> elem;
        }
    }

    for (auto& row : B) {
        for (auto& elem : row) {
            std::cin >> elem;
        }
    }

    std::vector<std::vector<double>> C = matrix_multiply_avx2(A, B);

    for (const auto& row : C) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
