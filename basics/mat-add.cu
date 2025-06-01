#include <cuda_runtime.h>
#include <stdio.h>

#define N 4 // Size of the matrix (4x4)

// CUDA kernel for matrix addition using shared memory
__global__ void matrixAdd(int *a, int *b, int *c) {
    // Declare shared memory for small chunks of matrices
    __shared__ int s_a[N][N];
    __shared__ int s_b[N][N];

    // Calculate thread's position in the matrix
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Load data from global to shared memory
    s_a[row][col] = a[row * N + col];
    s_b[row][col] = b[row * N + col];

    // Ensure all threads have loaded data
    __syncthreads();

    // Perform addition in shared memory
    int sum = s_a[row][col] + s_b[row][col];

    // Write result to global memory
    c[row * N + col] = sum;
}

int main() {
    // Size of arrays
    int size = N * N * sizeof(int);

    // Host (CPU) arrays
    int h_a[N][N] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    int h_b[N][N] = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};
    int h_c[N][N];

    // Device (GPU) arrays
    int *d_a, *d_b, *d_c;

    // Allocate memory on GPU
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from CPU to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define block and grid size
    dim3 blockSize(N, N); // 4x4 threads per block
    dim3 gridSize(1, 1);  // 1 block

    // Launch kernel
    matrixAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c);

    // Copy result back to CPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print result
    printf("Result Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_c[i][j]);
        }
        printf("\n");
    }

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}