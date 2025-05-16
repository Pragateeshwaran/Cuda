#include <iostream>
#include <cuda_runtime.h>

__global__ void printWarpSize() {
    // Print the warp size to the console
    printf("Warp size is: %d\n", warpSize);
}

int main() {
    // Launch a kernel with a grid of 1 block and 32 threads
    printWarpSize<<<1, 32>>>();
    
    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    
    return 0;
}
