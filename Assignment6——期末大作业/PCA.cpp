#include <iostream>
#include <chrono>
#include <CL/sycl.hpp>
#include <random>
#include <vector>
#include <algorithm>

void pcaGPU(std::vector<float>& data, std::vector<float>& result, int numSamples, int numFeatures) {
    sycl::queue queue(sycl::gpu_selector{});

    const int localSize = 128;  // local work group size, should be tuned for your specific device
    const int numGroups = (numSamples + localSize - 1) / localSize;

    sycl::buffer<float, 1> dataBuffer(data.data(), sycl::range<1>(numSamples * numFeatures));
    sycl::buffer<float, 1> resultBuffer(result.data(), sycl::range<1>(numFeatures));

    queue.submit([&](sycl::handler& cgh) {
        auto dataAccessor = dataBuffer.get_access<sycl::access::mode::read_write>(cgh);  // 修改为read_write模式
        auto resultAccessor = resultBuffer.get_access<sycl::access::mode::write>(cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> localMemory(sycl::range<1>(localSize), cgh);

        cgh.parallel_for<class pca>(
            sycl::nd_range<1>(sycl::range<1>(numGroups * localSize), sycl::range<1>(localSize)),
            [=](sycl::nd_item<1> item) {
                int localId = item.get_local_id()[0];
                int groupId = item.get_group(0);
                int globalId = item.get_global_id(0);

                // Step 1: Calculate the mean of each feature
                float sum = 0.0f;
                for (int i = localId; i < numSamples; i += localSize) {
                    for (int j = 0; j < numFeatures; j++) {
                        sum += dataAccessor[i * numFeatures + j];
                    }
                }
                localMemory[localId] = sum;

                item.barrier(sycl::access::fence_space::local_space);

                // Reduce the partial sums
                for (int stride = localSize / 2; stride > 0; stride /= 2) {
                    if (localId < stride) {
                        localMemory[localId] += localMemory[localId + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Write the final result to global memory
                if (localId == 0) {
                    resultAccessor[groupId] = localMemory[0] / numSamples;
                }

                item.barrier(sycl::access::fence_space::local_space);

                // Step 2: Subtract the mean from each feature
                float mean = resultAccessor[groupId];
                for (int i = localId; i < numSamples; i += localSize) {
                    for (int j = 0; j < numFeatures; j++) {
                        dataAccessor[i * numFeatures + j] -= mean;
                    }
                }
            }
        );
        });
    queue.wait_and_throw();
}


// PCA的CPU实现
void pcaCPU(const std::vector<float>& data, std::vector<float>& result, int numSamples, int numFeatures) {
    std::vector<float> featureMeans(numFeatures, 0.0f);

    // Step 1: Calculate the mean of each feature
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            featureMeans[j] += data[i * numFeatures + j];
        }
    }
    for (int j = 0; j < numFeatures; j++) {
        featureMeans[j] /= numSamples;
    }

    // Step 2: Subtract the mean from each feature
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            result[i * numFeatures + j] = data[i * numFeatures + j] - featureMeans[j];
        }
    }
}

int main() {
    const int numSamples = 100000;
    const int numFeatures = 1000;

    std::vector<float> data(numSamples * numFeatures);
    std::vector<float> resultGPU(numFeatures);
    std::vector<float> resultCPU(numSamples * numFeatures);

    // Initialize input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < numSamples * numFeatures; i++) {
        data[i] = dis(gen);
    }

    // Measure GPU time
    auto startTimeGPU = std::chrono::high_resolution_clock::now();
    pcaGPU(data, resultGPU, numSamples, numFeatures);
    auto endTimeGPU = std::chrono::high_resolution_clock::now();
    auto durationGPU = std::chrono::duration<double, std::ratio<1, 1>>(endTimeGPU - startTimeGPU).count();

    // Measure CPU time
    auto startTimeCPU = std::chrono::high_resolution_clock::now();
    pcaCPU(data, resultCPU, numSamples, numFeatures);
    auto endTimeCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration<double, std::ratio<1, 1>>(endTimeCPU - startTimeCPU).count();

    std::cout << "GPU Time: " << durationGPU << " seconds" << std::endl;
    std::cout << "CPU Time: " << durationCPU << " seconds" << std::endl;

    // Cleanup resources
    data.clear();
    resultGPU.clear();
    resultCPU.clear();

    return 0;
}
