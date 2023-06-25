#include <chrono>
#include <iostream>
#include <CL/sycl.hpp>
#include <cmath>

using namespace std;
using namespace sycl;
using namespace chrono;

// GPU kernel for PCA
void gpu_pca(float* input, float* output, int numSamples, int numFeatures, int numComponents, sycl::queue& q) {
    auto start = high_resolution_clock::now();
    // Create device buffers for input and output data
    buffer<float, 1> inputBuffer(input, range<1>(numSamples * numFeatures));
    buffer<float, 1> outputBuffer(output, range<1>(numSamples * numComponents));

    // Determine the work group size
    const int workGroupSize = 64;

    // Submit GPU kernel
    auto e = q.submit([&](sycl::handler& h) {
        auto inputAccessor = inputBuffer.get_access<access::mode::read>(h);
        auto outputAccessor = outputBuffer.get_access<access::mode::write>(h);

        h.parallel_for<class pca_kernel>(range<1>(numSamples), [=](id<1> index) {
            // Perform PCA calculation
            float sum = 0.0f;
            for (int i = 0; i < numFeatures; i++) {
                sum += inputAccessor[index[0] * numFeatures + i];
            }
            float mean = sum / numFeatures;

            float variance = 0.0f;
            for (int i = 0; i < numFeatures; i++) {
                float diff = inputAccessor[index[0] * numFeatures + i] - mean;
                variance += diff * diff;
            }
            variance /= numFeatures;

            for (int i = 0; i < numComponents; i++) {
                float value = 0.0f;
                for (int j = 0; j < numFeatures; j++) {
                    value += (inputAccessor[index[0] * numFeatures + j] - mean) *
                        (inputAccessor[index[0] * numFeatures + j] - mean);
                }
                outputAccessor[index[0] * numComponents + i] = value / sqrt(variance);
            }
            });
        });
    e.wait();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << "GPU Computation Time: " << duration << " ms" << std::endl;
}


// CPU implementation of PCA
void cpu_pca(float* input, float* output, int numSamples, int numFeatures, int numComponents) {
    auto start = high_resolution_clock::now();

    for (int i = 0; i < numSamples; i++) {
        // Perform PCA calculation
        float sum = 0.0f;
        for (int j = 0; j < numFeatures; j++) {
            sum += input[i * numFeatures + j];
        }
        float mean = sum / numFeatures;

        float variance = 0.0f;
        for (int j = 0; j < numFeatures; j++) {
            float diff = input[i * numFeatures + j] - mean;
            variance += diff * diff;
        }
        variance /= numFeatures;

        for (int j = 0; j < numComponents; j++) {
            float value = 0.0f;
            for (int k = 0; k < numFeatures; k++) {
                value += (input[i * numFeatures + k] - mean) * (input[i * numFeatures + k] - mean);
            }
            output[i * numComponents + j] = value / sqrt(variance);
        }
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << "CPU Computation Time: " << duration << " ms" << std::endl;
}

int main() {
    const int numSamples = 8000;
    const int numFeatures = 1024;
    const int numComponents = 128;

    // Allocate memory for input and output data
    float* input = new float[numSamples * numFeatures];
    float* output_gpu = new float[numSamples * numComponents];
    float* output_cpu = new float[numSamples * numComponents];

    // Initialize input data
    for (int i = 0; i < numSamples * numFeatures; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Create SYCL queue for GPU execution
    queue q(gpu_selector{});

    // Perform PCA on GPU
    gpu_pca(input, output_gpu, numSamples, numFeatures, numComponents, q);

    // Perform PCA on CPU
    cpu_pca(input, output_cpu, numSamples, numFeatures, numComponents);

    // Cleanup
    delete[] input;
    delete[] output_gpu;
    delete[] output_cpu;

    return 0;
}
