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

// Function to get the number of Tensor Cores based on the GPU architecture
int getTensorCoresCount(const cudaDeviceProp& deviceProp) {
    switch (deviceProp.major) {
        case 7: // Volta
            return deviceProp.multiProcessorCount * 640; // 640 Tensor Cores per SM on Volta
        case 8: // Turing and Ampere
            return deviceProp.multiProcessorCount * 512; // 512 Tensor Cores per SM on Turing/Ampere
        default:
            return 0; // Tensor Cores not available
    }
}

// Function to get the number of RT Cores based on the GPU architecture
int getRTcoresCount(const cudaDeviceProp& deviceProp) {
    if (deviceProp.major >= 7 && deviceProp.major <= 8) { // Turing and Ampere
        return deviceProp.multiProcessorCount; // 1 RT Core per SM on Turing/Ampere
    }
    return 0; // RT Cores not available
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

        // Get the number of Tensor Cores
        int tensorCoresCount = getTensorCoresCount(deviceProp);
        std::cout << "Tensor Cores Count: " << tensorCoresCount << std::endl;

        // Get the number of RT Cores
        int rtCoresCount = getRTcoresCount(deviceProp);
        std::cout << "RT Cores Count: " << rtCoresCount << std::endl;
    }

    return 0;
}

// git rm --cached cuda_12.6.2_560.35.03_linux.run
