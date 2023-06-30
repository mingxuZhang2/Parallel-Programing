#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <pthread.h>
#include <immintrin.h>

// Thread parameters structure
struct ThreadParams {
    int start;                                  // Starting index for this thread
    int end;                                    // Ending index for this thread
    const std::vector<std::vector<int>>* input; // Input to be processed
    std::vector<std::vector<double>>* output;   // Output to be written to
    const std::vector<std::vector<double>>* weights; // Weights for the embedding layer
    int embeddingSize;                          // Embedding size
};

// Thread function
void* processBatch(void* args) {
    // Cast void* to ThreadParams*
    ThreadParams* params = static_cast<ThreadParams*>(args);

    for (int b = params->start; b < params->end; ++b) {
        for (size_t i = 0; i < (*params->input)[b].size(); ++i) {
            __m256d sum_vec = _mm256_setzero_pd();

            for (int j = 0; j < params->embeddingSize; ++j) {
                __m256d input_vec = _mm256_set1_pd((*params->weights)[(*params->input)[b][i]][j]);

                sum_vec = _mm256_add_pd(sum_vec, input_vec);
            }

            _mm256_store_pd(&(*params->output)[b][i], sum_vec);
        }
    }

    return nullptr;
}

// 前向传播函数 (SIMD和Pthread优化)
std::vector<std::vector<double>> forwardBatch(const std::vector<std::vector<int>>& batchInput, const std::vector<std::vector<double>>& weights, int embeddingSize, int numThreads) {
    int batchSize = batchInput.size();

    std::vector<std::vector<double>> output(batchSize, std::vector<double>(embeddingSize, 0.0));

    std::vector<pthread_t> threads(numThreads);
    std::vector<ThreadParams> threadParams(numThreads);

    // Create threads
    for (int i = 0; i < numThreads; ++i) {
        threadParams[i] = { i * batchSize / numThreads, (i + 1) * batchSize / numThreads, &batchInput, &output, &weights, embeddingSize };
        pthread_create(&threads[i], nullptr, processBatch, &threadParams[i]);
    }

    // Wait for threads to finish
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    return output;
}

// Embedding class
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

    std::vector<std::vector<double>> forward(const std::vector<std::vector<int>>& input, int numThreads) {
        return forwardBatch(input, weights, embeddingSize, numThreads);
    }
};

int main() {
    int inputSize = 1024;        // 输入大小
    int embeddingSize = 512;     // 嵌入维度
    int batchSize = 4;           // 批处理大小
    int numThreads = 4;          // 线程数量

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
    std::vector<std::vector<double>> output = embedding.forward(batchInput, numThreads);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    // 输出结果
    std::cout << "Total embedding time: " << elapsedTime.count() << " seconds" << std::endl;
    return 0;
}
