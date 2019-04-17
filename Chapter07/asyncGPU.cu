#include <omp.h>
#include <stdio.h>

__global__ void fillKernel(int *a, int n, int offset)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n) {
    register int delay=1000000;
    while(delay > 0) delay--;
    a[tid] = delay + offset+tid;
  }
}

int main(int argc, char* argv[])
{
  int nStreams=5;
  int n = 1024;
  int size = n * sizeof(int);
  cudaStream_t streams[nStreams];
  int *d_A[nStreams];

  for(int i=0; i < nStreams; i++) {
    cudaMalloc(&d_A[i],size);
    if(cudaStreamCreate(&streams[i]) != 0) {
      fprintf(stderr,"Stream create failed!\n"); exit(1);
    }
  }

  int *h_A;
  cudaHostAlloc(&h_A, nStreams*size, cudaHostAllocPortable);

  int nThreadsPerBlock= 512;
  int nBlocks= n/nThreadsPerBlock + ((n%nThreadsPerBlock)?1:0);
  double startTime = omp_get_wtime();
  for(int i=0; i < nStreams; i++) {
#ifdef USE_SINGLE_STREAM
    fillKernel<<<nBlocks, nThreadsPerBlock, 0>>>(d_A[i], n, i*n);
#else
    fillKernel<<<nBlocks, nThreadsPerBlock, 0, streams[i]>>>(d_A[i], n, i*n);
#endif
  }
  cudaDeviceSynchronize();
  double endTime= omp_get_wtime();
  printf("runtime %f\n",endTime-startTime);
  for(int i=0; i < nStreams; i++) {
    cudaMemcpyAsync(&h_A[i*n], d_A[i], size, cudaMemcpyDefault, streams[i]);
  }
  cudaDeviceSynchronize();

  for(int i=0; i < nStreams*n; i++)
    if(h_A[i] != i) {
      printf("Error h_A[%d] = %d\n",i,h_A[i]); exit(1);
    }
  printf("Success!\n");

  for(int i=0; i < nStreams; i++) {
    cudaFree(d_A[i]);
  }
  return(0);
}
