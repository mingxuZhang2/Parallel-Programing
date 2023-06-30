#include <iostream>
#include <chrono>
#include <CL/sycl.hpp>
#include <random>
// 卷积操作的GPU实现

void convolutionGPU(const float* input, const float* kernel, float* output, int inputSize, int kernelSize) {
    sycl::queue queue(sycl::gpu_selector{});

    const int localSize = 256   ;  // local work group size, should be tuned for your specific device
    const int numGroups = (inputSize + localSize - 1) / localSize;

    sycl::buffer<float, 1> inputBuffer(input, sycl::range<1>(inputSize));
    sycl::buffer<float, 1> kernelBuffer(kernel, sycl::range<1>(kernelSize));
    sycl::buffer<float, 1> outputBuffer(output, sycl::range<1>(inputSize));

    queue.submit([&](sycl::handler& cgh) {
        auto inputAccessor = inputBuffer.get_access<sycl::access::mode::read>(cgh);
        auto kernelAccessor = kernelBuffer.get_access<sycl::access::mode::read>(cgh);
        auto outputAccessor = outputBuffer.get_access<sycl::access::mode::write>(cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> localMemory(sycl::range<1>(localSize + kernelSize - 1), cgh);

        cgh.parallel_for<class convolution>(
            sycl::nd_range<1>(sycl::range<1>(numGroups * localSize), sycl::range<1>(localSize)),
            [=](sycl::nd_item<1> item) {
                int localId = item.get_local_id()[0];
                int groupId = item.get_group(0);
                int globalId = item.get_global_id(0);

                if (localId < kernelSize) {
                    localMemory[localId] = (groupId * localSize + localId < inputSize) ? inputAccessor[groupId * localSize + localId] : 0;
                }

                localMemory[localId + kernelSize - 1] = (groupId * localSize + localId + kernelSize - 1 < inputSize) ? inputAccessor[groupId * localSize + localId + kernelSize - 1] : 0;

                item.barrier(sycl::access::fence_space::local_space);

                float result = 0;

                for (int j = 0; j < kernelSize; j++) {
                    result += localMemory[localId + j] * kernelAccessor[j];
                }

                if (globalId < inputSize) {
                    outputAccessor[globalId] = result;
                }
            }
        );
        });
    queue.wait_and_throw();
}

// 卷积操作的CPU实现
void convolutionCPU(const float* input, const float* kernel, float* output, int inputSize, int kernelSize) {
    for (int i = 0; i < inputSize; i++) {
        output[i] = 0;
        for (int j = 0; j < kernelSize; j++) {
            if (i - j >= 0 && i - j < inputSize) {
                output[i] += input[i - j] * kernel[j];
            }
        }
    }
}

int main() {
    const int inputSize = 250000;
    const int kernelSize = 20000;

    float* input = new float[inputSize];
    float* kernel = new float[kernelSize];
    float* outputGPU = new float[inputSize];
    float* outputCPU = new float[inputSize];

    // 初始化输入数据和卷积核
        // 生成随机数填充输入和卷积核
    std::random_device rd;  //用于获取种子数据
    std::mt19937 gen(rd()); //标准 mersenne_twister_engine
    std::uniform_real_distribution<> dis(0.0, 1.0); //设定随机数分布范围

    for (int i = 0; i < inputSize; i++) {
        input[i] = dis(gen);
    }
    for (int i = 0; i < kernelSize; i++) {
        kernel[i] = dis(gen);
    }
    // 测量GPU时间
    auto startTimeGPU = std::chrono::high_resolution_clock::now();
    convolutionGPU(input, kernel, outputGPU, inputSize, kernelSize);
    auto endTimeGPU = std::chrono::high_resolution_clock::now();
    auto durationGPU = std::chrono::duration<double, std::ratio<1, 1>>(endTimeGPU - startTimeGPU).count();

    // 测量CPU时间
    auto startTimeCPU = std::chrono::high_resolution_clock::now();
    convolutionCPU(input, kernel, outputCPU, inputSize, kernelSize);
    auto endTimeCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration<double, std::ratio<1, 1>>(endTimeCPU - startTimeCPU).count();

    std::cout << "GPU Time: " << durationGPU << " seconds" << std::endl;
    std::cout << "CPU Time: " << durationCPU << " seconds" << std::endl;

    // 清理资源
    delete[] input;
    delete[] kernel;
    delete[] outputGPU;
    delete[] outputCPU;

    return 0;
}
