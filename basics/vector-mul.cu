#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    float *d_array;
    cudaError_t err = cudaMalloc(&d_array, 1024 * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaFree(d_array);
    return 0;
}