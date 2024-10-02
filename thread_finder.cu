#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "Total SMs: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "CUDA Cores per SM: " << deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsDim[0] << std::endl;
        std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        
        // Total number of threads that can run simultaneously
        int totalThreads = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor;
        std::cout << "Total Threads (per launch): " << totalThreads << std::endl;
    }

    return 0;
}
