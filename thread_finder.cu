#include <iostream>
#include <cuda_runtime.h>

// Function to get the number of CUDA cores per SM based on the GPU architecture
int getCudaCoresPerSM(const cudaDeviceProp& deviceProp) {
    // The number of CUDA cores per SM for different architectures
    switch (deviceProp.major) {
        case 2: // Fermi
            return 32; // 32 cores per SM
        case 3: // Kepler
            return 192; // 192 cores per SM
        case 5: // Maxwell
            return 128; // 128 cores per SM
        case 6: // Pascal
            return 64; // 64 cores per SM
        case 7: // Volta and Turing
            return 64; // 64 cores per SM
        case 8: // Ampere
            return 128; // 128 cores per SM
        default:
            return 0; // Unknown architecture
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "Total SMs: " << deviceProp.multiProcessorCount << std::endl;

        // Get the number of CUDA cores per SM based on the architecture
        int cudaCoresPerSM = getCudaCoresPerSM(deviceProp);
        std::cout << "CUDA Cores per SM: " << cudaCoresPerSM << std::endl;

        std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        
        // Total number of threads that can run simultaneously
        int totalThreads = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor;
        std::cout << "Total Threads (per launch): " << totalThreads << std::endl;
    }

    return 0;
}

// Device 0: NVIDIA GeForce RTX 3060
// Total SMs: 28
// CUDA Cores per SM: 128
// Max Threads per Block: 1024
// Total Threads (per launch): 43008
