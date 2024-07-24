#include <stdio.h>

// CUDA kernel to print "Hello, World!" from each thread
__global__ void helloFromGPU() {
    // Calculate the thread's unique ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print "Hello, World!" from each thread
    printf("Hello, World! from thread %d\n", tid);
}

int main() {
    // Define grid and block dimensions
    int numBlocks = 1;    // Number of blocks
    int blockSize = 10;   // Threads per block

    // Launch the CUDA kernel with specified grid and block dimensions
    helloFromGPU<<<numBlocks, blockSize>>>();

    // Synchronize threads after kernel execution
    cudaDeviceSynchronize();

    // Check for any errors during the kernel launch
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    return 0;
}

