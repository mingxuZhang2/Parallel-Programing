#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <random>
#include <pthread.h>
#pragma comment(lib, "pthreadVC2.lib")
struct MatrixData {
    const std::vector<float>* a;
    const std::vector<float>* b;
    std::vector<float>* c;
    int m;
    int n;
    int p;
    int start_row;
    int end_row;
};

void* matrix_multiply_avx2_thread(void* arg) {
    MatrixData* data = static_cast<MatrixData*>(arg);
    const std::vector<float>& a = *data->a;
    const std::vector<float>& b = *data->b;
    std::vector<float>& c = *data->c;
    int m = data->m;
    int n = data->n;
    int p = data->p;
    int start_row = data->start_row;
    int end_row = data->end_row;

    for (int i = start_row; i < end_row; ++i) {
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

    return nullptr;
}

int main() {
    int m = 500;
    int n = 500;
    int p = 500;
    int num_threads = 4;

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

    // 创建线程
    auto start = std::chrono::steady_clock::now();
    std::vector<pthread_t> threads(num_threads);
    std::vector<MatrixData> thread_data(num_threads);

    int chunk_size = m / num_threads;
    // 创建并启动线程
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].a = &a;
        thread_data[i].b = &b;
        thread_data[i].c = &c;
        thread_data[i].m = m;
        thread_data[i].n = n;
        thread_data[i].p = p;
        thread_data[i].start_row = i * chunk_size;
        thread_data[i].end_row = (i == num_threads - 1) ? m : (i + 1) * chunk_size;

        pthread_create(&threads[i], nullptr, matrix_multiply_avx2_thread, &thread_data[i]);
    }
    // 等待所有线程完成
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "矩阵乘法完成" << std::endl;
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "计算耗时: " << elapsed.count() << " 秒" << std::endl;
    return 0;
}