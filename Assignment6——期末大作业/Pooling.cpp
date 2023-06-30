#include <iostream>
#include <chrono>
#include <CL/sycl.hpp>
#include <random>

// 池化操作的GPU实现
void poolingGPU(const float* input, float* output, int inputSize, int filterSize) {
    sycl::queue queue(sycl::gpu_selector{});

    const int localSize = 256;  // local work group size, should be tuned for your specific device
    const int numGroups = (inputSize + localSize - 1) / localSize;

    sycl::buffer<float, 1> inputBuffer(input, sycl::range<1>(inputSize));
    sycl::buffer<float, 1> outputBuffer(output, sycl::range<1>(inputSize / filterSize));

    queue.submit([&](sycl::handler& cgh) {
        auto inputAccessor = inputBuffer.get_access<sycl::access::mode::read>(cgh);
        auto outputAccessor = outputBuffer.get_access<sycl::access::mode::write>(cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> localMemory(sycl::range<1>(localSize), cgh);

        cgh.parallel_for<class pooling>(
            sycl::nd_range<1>(sycl::range<1>(numGroups * localSize), sycl::range<1>(localSize)),
            [=](sycl::nd_item<1> item) {
                int localId = item.get_local_id()[0];
                int groupId = item.get_group(0);
                int globalId = item.get_global_id(0);

                localMemory[localId] = (groupId * localSize + localId < inputSize) ? inputAccessor[groupId * localSize + localId] : 0;

                item.barrier(sycl::access::fence_space::local_space);

                float result = localMemory[localId];
                for (int i = 1; i < filterSize; i++) {
                    int index = groupId * localSize + localId + i;
                    if (index < inputSize && localMemory[localId] < localMemory[index]) {
                        result = localMemory[index];
                    }
                }

                if (globalId < inputSize / filterSize) {
                    outputAccessor[globalId] = result;
                }
            }
        );
        });
    queue.wait_and_throw();
}

// 池化操作的CPU实现
void poolingCPU(const float* input, float* output, int inputSize, int filterSize) {
    for (int i = 0; i < inputSize / filterSize; i++) {
        output[i] = input[i * filterSize];
        for (int j = 1; j < filterSize; j++) {
            if (input[i * filterSize + j] > output[i]) {
                output[i] = input[i * filterSize + j];
            }
        }
    }
}

int main() {
    const int inputSize = 32000*32000;
    const int filterSize = 128;

    float* input = new float[inputSize];
    float* outputGPU = new float[inputSize / filterSize];
    float* outputCPU = new float[inputSize / filterSize];

    // 初始化输入数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < inputSize; i++) {
        input[i] = dis(gen);
    }

    // 测量GPU时间
    auto startTimeGPU = std::chrono::high_resolution_clock::now();
    poolingGPU(input, outputGPU, inputSize, filterSize);
    auto endTimeGPU = std::chrono::high_resolution_clock::now();
    auto durationGPU = std::chrono::duration<double, std::ratio<1, 1>>(endTimeGPU - startTimeGPU).count();

    // 测量CPU时间
    auto startTimeCPU = std::chrono::high_resolution_clock::now();
    poolingCPU(input, outputCPU, inputSize, filterSize);
    auto endTimeCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration<double, std::ratio<1, 1>>(endTimeCPU - startTimeCPU).count();

    std::cout << "GPU Time: " << durationGPU << " seconds" << std::endl;
    std::cout << "CPU Time: " << durationCPU << " seconds" << std::endl;

    // 清理资源
    delete[] input;
    delete[] outputGPU;
    delete[] outputCPU;

    return 0;
}
