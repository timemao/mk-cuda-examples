#include <stdio.h>
#ifndef REDUCE_H
#define REDUCE_H

//Define the number of blocks as a multiple of the number of SM
// and the number of threads as the maximum resident on the SM
#define N_BLOCKS (1*14)
#define N_THREADS 1024
#define WARP_SIZE 32

template <class T, typename UnaryFunction, typename BinaryFunction>
  __global__ void
  _functionReduce(T *g_odata, unsigned int n, T initVal,
		  UnaryFunction fcn, BinaryFunction fcn1)
{
  T myVal = initVal;
  
  { // 1) Use fastest memory first. 
    const int gridSize = blockDim.x*gridDim.x;
    //for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridSize)
      //myVal = fcn1(fcn(i), myVal);
    for(int i = n-1 -(blockIdx.x * blockDim.x + threadIdx.x);
	i >= 0; i -= gridSize)
      myVal = fcn1(fcn(i), myVal);
  }
  
  // 2) Use the second fastest memory (shared memory) in a warp
  // synchronous fashion.
  // Create shared memory for per-block reduction. 
  // Reuse the registers in the first warp.
  volatile __shared__ T smem[N_THREADS-WARP_SIZE];
  
  // put all the register values into a shared memory
  if(threadIdx.x >= WARP_SIZE) smem[threadIdx.x - WARP_SIZE] = myVal;
  __syncthreads(); // wait for all threads in the block to complete.

  if(threadIdx.x < WARP_SIZE) {
    // now using just one warp. The SM can only run one warp at a time
#pragma unroll 
    for(int i=threadIdx.x; i < (N_THREADS-WARP_SIZE); i += WARP_SIZE)
      myVal = fcn1(myVal,(T)smem[i]);
    smem[threadIdx.x] = myVal; // save myVal in this warp to the start of smem
  }
  // reduce shared memory.
  if (threadIdx.x < 16)
    smem[threadIdx.x] = fcn1((T)smem[threadIdx.x],(T)smem[threadIdx.x + 16]);
  if (threadIdx.x < 8)
    smem[threadIdx.x] = fcn1((T)smem[threadIdx.x],(T)smem[threadIdx.x + 8]);
  if (threadIdx.x < 4)
    smem[threadIdx.x] = fcn1((T)smem[threadIdx.x],(T)smem[threadIdx.x + 4]);
  if (threadIdx.x < 2)
    smem[threadIdx.x] = fcn1((T)smem[threadIdx.x],(T)smem[threadIdx.x + 2]);
  if (threadIdx.x < 1)
    smem[threadIdx.x] = fcn1((T)smem[threadIdx.x],(T)smem[threadIdx.x + 1]);

  // 3) Use global memory as a last resort to transfer results to the host
  // write result for each block to global mem 
  if (threadIdx.x == 0) g_odata[blockIdx.x] = smem[0];
  // Can put the final reduction across SM here if desired.
}

template<typename T, typename UnaryFunction, typename BinaryFunction>
  inline void partialReduce(const int n, T** d_partialVals, T initVal,
			    UnaryFunction const& fcn,
			    BinaryFunction const& fcn1)
{
  if(*d_partialVals == NULL)
    cudaMalloc(d_partialVals, (N_BLOCKS+1) * sizeof(T));

  _functionReduce<T><<< N_BLOCKS, N_THREADS>>>(*d_partialVals, n,
						initVal, fcn, fcn1);
}
template<typename T, typename UnaryFunction, typename BinaryFunction>
  inline T functionReduce(const int n, T** d_partialVals, T initVal,
			   UnaryFunction const& fcn,
			   BinaryFunction const& fcn1)
{
  partialReduce(n, d_partialVals, initVal, fcn, fcn1);

  //Get the values onto the host
  //Note: uses the default stream in the current context
  T h_partialVals[N_BLOCKS];

  if(cudaMemcpy(h_partialVals, *d_partialVals, sizeof(T)*N_BLOCKS, 
		cudaMemcpyDeviceToHost) != cudaSuccess) {
    cerr << "_functionReduce copy failed!" << endl; 
    exit(1);
  }

  // Perform the final reduction
  T val = h_partialVals[0];
  for(int i=1; i < N_BLOCKS; i++) val = fcn1(h_partialVals[i],val);
  return(val);
}
#endif
