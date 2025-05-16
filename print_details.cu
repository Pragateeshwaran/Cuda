#include <iostream>
#include <cuda_runtime.h>

int main() {
    int N = 100000;  // Total number of elements to process
    int blockSize = 256;  // Threads per block (commonly 128, 256, or 512)

    // Calculate grid size
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Calculate total threads
    int totalThreads = gridSize * blockSize;
    
    // Calculate total warps
    int totalWarps = totalThreads / 32;

    // Output results
    std::cout << "Total Elements (N): " << N << std::endl;
    std::cout << "Threads per Block: " << blockSize << std::endl;
    std::cout << "Grid Size (Total Blocks): " << gridSize << std::endl;
    std::cout << "Total Threads Launched: " << totalThreads << std::endl;
    std::cout << "Total Warps: " << totalWarps << std::endl;

    return 0;
}

