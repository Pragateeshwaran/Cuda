#include <stdio.h>

// CUDA Kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
    printf("%d ", C[i]);    
}

int main() {
    int N = 1 << 20; // 1 Million elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize vectors and calculate their sums
    float sum_A = 0, sum_B = 0;
    for(int i = 0; i < N; i++) {
        h_A[i] = float(i);
        h_B[i] = float(i * 2);
        sum_A += h_A[i];
        sum_B += h_B[i];
    }

    // Print the sums of the elements of A and B
    printf("Sum of vector A: %f\n", sum_A);
    printf("Sum of vector B: %f\n", sum_B);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for(int i = 0; i < N; i++) {
        if(h_C[i] != h_A[i] + h_B[i]) {
            printf("Error at index %d: %f != %f + %f\n", i, h_C[i], h_A[i], h_B[i]);
            break;
        }
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Vector addition completed successfully.\n");
    return 0;
}
