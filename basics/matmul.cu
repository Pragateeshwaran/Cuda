// #include <iostream>
// #include <cuda_runtime.h>

// #define M 2  // Rows of A
// #define K 3  // Columns of A and Rows of B
// #define N 2  // Columns of B

// __global__ void matrixMulKernel(float* A, float* B, float* C, int m, int k, int n) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y; // i
//     int col = blockIdx.x * blockDim.x + threadIdx.x; // j

//     if (row < m && col < n) {
//         float sum = 0.0;
//         for (int i = 0; i < k; i++) {
//             sum += A[row * k + i] * B[i * n + col];
//         }
//         C[row * n + col] = sum;
//     }
// }

// void printMatrix(float* mat, int rows, int cols) {
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             std::cout << mat[i * cols + j] << " ";
//         }
//         std::cout << "\n";
//     }
// }

// int main() {
//     // Host matrices
//     float h_A[M*K] = {1, 2, 3, 4, 5, 6};
//     float h_B[K*N] = {7, 8, 9, 10, 11, 12};
//     float h_C[M*N];

//     // Device matrices
//     float *d_A, *d_B, *d_C;
//     cudaMalloc((void**)&d_A, M*K*sizeof(float));
//     cudaMalloc((void**)&d_B, K*N*sizeof(float));
//     cudaMalloc((void**)&d_C, M*N*sizeof(float));

//     // Copy host to device
//     cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

//     // Kernel launch config
//     dim3 threadsPerBlock(16, 16);
//     dim3 blocksPerGrid((N+15)/16, (M+15)/16);
//     matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

//     // Copy result back to host
//     cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

//     // Print result
//     std::cout << "Matrix A:\n"; printMatrix(h_A, M, K);
//     std::cout << "Matrix B:\n"; printMatrix(h_B, K, N);
//     std::cout << "Matrix C = A x B:\n"; printMatrix(h_C, M, N);

//     // Free memory
//     cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

//     return 0;
// }



#include<iostream>
#include<cuda_runtime.h>
using namespace std;
__global__ void matmul(float *A, float *B, float *C, int M, int K, int N) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;  
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;  
    if(idx_x< N && idx_y < M) {
        float sums = 0.0f;
        for(int i=0; i < K; i++){
            sums += A[idx_y * K + i] * B[i * N + idx_x];
        }
         C[idx_y*N + idx_x] = sums;
    }
   
}

int main(){
    float *A, *B, *C;
    int M = 2;
    int N = 2;
    int K = 5;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    A = (float*)malloc(size_A);
    B = (float*)malloc(size_B);
    C = (float*)malloc(size_C);
    for(int i = 0; i < M * K; i++) {
        A[i] = i;
    }
    for(int i = 0; i < K * N; i++) {
        B[i] = i;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    cout << "Result Matrix C:\n";
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            cout << C[i * N + j] << " ";
        }
        cout << endl;
    }

}