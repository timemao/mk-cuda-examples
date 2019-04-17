//seqRuntime.cu
#include <iostream>
using namespace std;

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void fillKernel(int *a, int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n) a[tid] = tid;
}

void fill(int* d_a, int n)
{
  int nThreadsPerBlock= 512;
  int nBlocks= n/nThreadsPerBlock + ((n%nThreadsPerBlock)?1:0);

  fillKernel <<< nBlocks, nThreadsPerBlock >>> (d_a, n);
} 

int main()
{
  const int N=50000;
  
  // task 1: create the array
  thrust::device_vector<int> a(N);
  
  // task 2: fill the array using the runtime
  fill(thrust::raw_pointer_cast(&a[0]),N);
  
  // task 3: calculate the sum of the array
  int sumA= thrust::reduce(a.begin(),a.end(), 0);
  
  // task 4: calculate the sum of 0 .. N-1
  int sumCheck=0;
  for(int i=0; i < N; i++) sumCheck += i;
  
  // task 5: check the results agree
  if(sumA == sumCheck) cout << "Test Succeeded!" << endl;
  else { cerr << "Test FAILED!" << endl; return(1);}
  
  return(0);
}


