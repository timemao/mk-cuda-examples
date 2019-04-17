#include <omp.h>
#include <stdio.h>
#include <thrust/reduce.h>

__global__ void fillKernel(int *a, int n, int offset)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n)
      a[tid] = offset+tid;
}

int main(int argc, char* argv[])
{
  int nGPU;
  int n = 1000000;
  int size = n * sizeof(int);
  
  cudaGetDeviceCount(&nGPU);
  
  int *d_A[nGPU];
  for(int i=0; i < nGPU; i++) {
    cudaSetDevice(i);
    if(cudaSetDeviceFlags(cudaDeviceScheduleYield)) {
      fprintf(stderr,"cudaSetDeviceFlags failed!\n"); exit(1);
    }
    cudaMalloc(&d_A[i],size);
  }
  
  int *h_A;
  cudaHostAlloc(&h_A, nGPU*size, cudaHostAllocMapped);
  
  for(int i=0; i < nGPU; i++) {
    int nThreadsPerBlock= 512;
    int nBlocks= n/nThreadsPerBlock + ((n%nThreadsPerBlock)?1:0);
    cudaSetDevice(i);
    fillKernel<<<nBlocks, nThreadsPerBlock>>>(d_A[i], n, i*n);
  }
  double sTime = omp_get_wtime();
  for(int i=0; i < nGPU; i++) {
    cudaSetDevice(i);
    cudaMemcpyAsync(&h_A[i*n], d_A[i], size, cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();
  double eTime = omp_get_wtime();
  printf("time %f\n", eTime-sTime);

  for(int i=0; i < nGPU*n; i++)
    if(h_A[i] != i) {
      printf("Error h_A[%d] = %d\n",i,h_A[i]); exit(1);
    }
  printf("Success!\n");

  cudaFreeHost(h_A);
  for(int i=0; i < nGPU; i++) {
    cudaSetDevice(i);
    cudaFree(d_A[i]);
  }
  return(0);
}
