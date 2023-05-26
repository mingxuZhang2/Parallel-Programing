#include <iostream>
#include <vector>
#include <random>
#include <chrono>
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

    // 前向传播函数
    std::vector<double> forward(const std::vector<int>& input) {
        std::vector<double> output(embeddingSize, 0.0);

        // 对输入进行嵌入
        for (int i : input) {
            for (int j = 0; j < embeddingSize; ++j) {
                output[j] += weights[i][j];
            }
        }

        return output;
    }
};

int main() {
    int inputSize = 100;        // 输入大小
    int embeddingSize = 50;     // 嵌入维度

    // 创建Embedding层对象
    Embedding embedding(inputSize, embeddingSize);

    // 随机生成输入
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, inputSize - 1);

    std::vector<int> input(10);
    for (int i = 0; i < 10; ++i) {
        input[i] = dis(gen);
    }
    auto startTime = std::chrono::high_resolution_clock::now();

    // 前向传播
    std::vector<double> output = embedding.forward(input);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    // 输出结果
    std::cout << "Total embedding time: " << elapsedTime.count() << " seconds" << std::endl;
    return 0;
}
