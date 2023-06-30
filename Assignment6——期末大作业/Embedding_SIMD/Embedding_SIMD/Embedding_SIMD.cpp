#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <immintrin.h>

// Embedding层的类定义
class Embedding {
private:
    int inputSize;       // 输入大小
    int embeddingSize;   // 嵌入维度
    std::vector<std::vector<double>> weights;  // 权重矩阵

public:
    Embedding(int inputSize, int embeddingSize) : inputSize(inputSize), embeddingSize(embeddingSize) {
        // 随机初始化权重矩阵
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        weights.resize(inputSize, std::vector<double>(embeddingSize));
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < embeddingSize; ++j) {
                weights[i][j] = dis(gen);
            }
        }
    }

    // 前向传播函数 (SIMD优化)
    std::vector<std::vector<double>> forwardBatch(const std::vector<std::vector<int>>& batchInput) {
        int batchSize = batchInput.size();
        int inputSize = batchInput[0].size();

        std::vector<std::vector<double>> output(batchSize, std::vector<double>(embeddingSize, 0.0));

        for (int b = 0; b < batchSize; b += 4) {
            for (int i = 0; i < inputSize; ++i) {
                __m256d sum_vec = _mm256_setzero_pd();

                for (int j = 0; j < embeddingSize; ++j) {
                    __m256d input_vec = _mm256_set_pd(
                        weights[batchInput[b + 3][i]][j],
                        weights[batchInput[b + 2][i]][j],
                        weights[batchInput[b + 1][i]][j],
                        weights[batchInput[b][i]][j]
                    );

                    sum_vec = _mm256_add_pd(sum_vec, input_vec);
                }

                _mm256_store_pd(&output[b][i], sum_vec);
            }
        }

        return output;
    }
};

int main() {
    int inputSize = 1024;        // 输入大小
    int embeddingSize = 512;     // 嵌入维度
    int batchSize = 4;           // 批处理大小

    // 创建Embedding层对象
    Embedding embedding(inputSize, embeddingSize);

    // 随机生成输入
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, inputSize - 1);

    std::vector<std::vector<int>> batchInput(batchSize, std::vector<int>(10));
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < 10; ++i) {
            batchInput[b][i] = dis(gen);
        }
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // 前向传播
    std::vector<std::vector<double>> output = embedding.forwardBatch(batchInput);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    // 输出结果
    std::cout << "Total embedding time: " << elapsedTime.count() << " seconds" << std::endl;
    return 0;
}
