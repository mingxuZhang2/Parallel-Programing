#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mpi.h>

std::vector<std::vector<double>> matrix_multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, int rows_per_proc) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();
    std::vector<std::vector<double>> C(rows_per_proc, std::vector<double>(p, 0.0));

    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

int main(int argc, char** argv) {
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

    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rows_per_proc = m / num_procs;
    int remaining_rows = m % num_procs;

    std::vector<std::vector<double>> local_A(rows_per_proc + (rank < remaining_rows ? 1 : 0), std::vector<double>(n));
    std::vector<std::vector<double>> local_C(rows_per_proc + (rank < remaining_rows ? 1 : 0), std::vector<double>(p));

    // 分发 A 到各个进程
    std::vector<int> sendcounts(num_procs, rows_per_proc);
    std::vector<int> displs(num_procs, 0);
    for (int i = 0; i < remaining_rows; i++) {
        sendcounts[i]++;
    }
    for (int i = 1; i < num_procs; i++) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    MPI_Scatterv(&A[0][0], &sendcounts[0], &displs[0], MPI_DOUBLE, &local_A[0][0], (rows_per_proc + 1) * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 广播 B 到所有进程
    MPI_Bcast(&B[0][0], n * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    local_C = matrix_multiply(local_A, B, rows_per_proc + (rank < remaining_rows ? 1 : 0));

    // 收集各个进程的计算结果到主进程
    MPI_Gatherv(&local_C[0][0], (rows_per_proc + (rank < remaining_rows ? 1 : 0)) * p, MPI_DOUBLE, &A[0][0], &sendcounts[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        std::cout << "矩阵乘法完成" << std::endl;
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "计算耗时: " << elapsed.count() << " 秒" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
