#include <CL/cl.h>
#include <stdio.h>
#define N 1024

const char* kernel_src = 
"__kernel void add(__global int* a, __global int* b, __global int* c) {"
"  int i = get_global_id(0);"
"  c[i] = a[i] + b[i];"
"}";

int main() {
    int a[N], b[N], c[N];
    for(int i = 0; i < N; i++) { a[i] = i; b[i] = i*2; }
    
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id devices[2];
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2, devices, NULL);
    
    cl_context ctx = clCreateContext(NULL, 2, devices, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(ctx, devices[1], 0, NULL);
    
    cl_mem buf_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N*sizeof(int), NULL, NULL);
    cl_mem buf_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N*sizeof(int), NULL, NULL);
    cl_mem buf_c = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N*sizeof(int), NULL, NULL);
    
    clEnqueueWriteBuffer(queue, buf_a, CL_TRUE, 0, N*sizeof(int), a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_b, CL_TRUE, 0, N*sizeof(int), b, 0, NULL, NULL);
    
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, NULL);
    clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(prog, "add", NULL);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_c);
    
    size_t global = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    
    clEnqueueReadBuffer(queue, buf_c, CL_TRUE, 0, N*sizeof(int), c, 0, NULL, NULL);
    
    printf("Result: %d + %d = %d\n", a[0], b[0], c[0]);
    
    clReleaseMemObject(buf_a); clReleaseMemObject(buf_b); clReleaseMemObject(buf_c);
    clReleaseKernel(kernel); clReleaseProgram(prog);
    clReleaseCommandQueue(queue); clReleaseContext(ctx);
    return 0;
}