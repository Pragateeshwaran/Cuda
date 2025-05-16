#include <iostream>
#include <cuda_runtime.h>

__global__ void display() {
    printf("Hello World from thread %d!\n", threadIdx.x);
}

int main() {
    display<<<2, 50>>>();  // Launch kernel with 1 block and 5 threads
    cudaDeviceSynchronize();  // Ensure all GPU tasks finish before exiting
    return 0;
}
