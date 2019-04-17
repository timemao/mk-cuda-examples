#include <stdio.h>

__global__ void fillKernel(int *a, int n, int offset)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n)
    a[tid+offset] = offset+tid;
}

int main(int argc, char* argv[])
{
  int nGPU;
  int n = 1000000;
  int size = n * sizeof(int);

  cudaGetDeviceCount(&nGPU);
  
  printf("nGPU %d\n",nGPU);
  int *h_A;
  cudaHostAlloc(&h_A, nGPU*size, cudaHostAllocMapped);

  for(int i=0; i < nGPU; i++) {
    int nThreadsPerBlock= 512;
    int nBlocks= n/nThreadsPerBlock + ((n%nThreadsPerBlock)?1:0);
    cudaSetDevice(i);
    fillKernel<<<nBlocks, nThreadsPerBlock>>>(h_A, n, i*n);
  }
  cudaDeviceSynchronize();

  for(int i=0; i < nGPU*n; i++)
    if(h_A[i] != i) {
      printf("Error h_A[%d] = %d\n",i,h_A[i]); exit(1);
    }
  printf("Success!\n");
  cudaFreeHost(h_A);
  return(0);
}
