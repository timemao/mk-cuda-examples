#include <iostream>
using namespace std;

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/random.h>

#include "functionReduce.h"

#ifndef RUNTYPE
#define RUNTYPE int
#endif
#ifndef FCN1
#define FCN1 plus
#endif
#include <iostream>
using namespace std;

template<class T1, class T2>
  struct prefetch : public thrust::unary_function<T1,T2> {
  const T1* data;
prefetch(T1* _data) : data(_data) {};
  __device__ 
    // This method prefetchs the previous grid of data point into the L2.
  T1 operator()(T2 i) {
    if( (i-N_BLOCKS*N_THREADS) > 0) { //prefetch the previous grid
      const T1 *pt = &data[i - (N_BLOCKS*N_THREADS)];
      asm volatile ("prefetch.global.L2 [%0];"::"l"(pt) );
    }
    return data[i];
  }
};

template<class T1, class T2>
  struct memFetch : public thrust::unary_function<T1,T2> {
  const T1* data;
  
 memFetch(T1* _data) : data(_data) {};
  __host__ __device__ 
    T1 operator()(T2 i) {
    return data[i];
  }
};

// a parallel random number generator
// http://groups.google.com/group/thrust-users/browse_thread/thread/dca23bfa678689a5
struct parallel_random_generator
{
  __host__ __device__
  unsigned int operator()(const unsigned int n) const
  {
    thrust::default_random_engine rng;
    // discard n numbers to avoid correlation
    rng.discard(n);
    // return a random number
    return rng() & 0x01;
  }
};

/************************************************************************/
/* The test routine                                                     */
/************************************************************************/

#define NTEST 100
template<typename T>
void doTest(const long nData, int op)
{
  T* d_partialVals=NULL;

  thrust::device_vector<T> d_data(nData);
  //fill d_data with random numbers between 0 … 1
  thrust::counting_iterator<int> index_sequence_begin(0);
  thrust::transform(index_sequence_begin,
		    index_sequence_begin + nData,
		    d_data.begin(), parallel_random_generator());
  cudaThreadSynchronize(); // wait for all the queued tasks to finish
  thrust::FCN1<T> fcn1;
  
  double startTime, endTime;
  T d_sum;
  T initVal = 0;
  switch(op) {
  case 0: {
    memFetch<T,int> fcn(thrust::raw_pointer_cast(&d_data[0]));
    startTime=omp_get_wtime();
    for(int loops=0; loops < NTEST; loops++)
      d_sum = functionReduce<T>(nData, &d_partialVals, initVal, fcn, fcn1);
    endTime=omp_get_wtime();
    cout << "NO prefetch ";
  } break;
  case 1: {
    prefetch<T,int> fcnPre(thrust::raw_pointer_cast(&d_data[0]));
    startTime=omp_get_wtime();
    for(int loops=0; loops < NTEST; loops++)
      d_sum = functionReduce<T>(nData, &d_partialVals, initVal, fcnPre, fcn1);
    endTime=omp_get_wtime();
    cout << "Using prefetch ";
  } break;
  default:
    startTime=omp_get_wtime();
    for(int loops=0; loops < NTEST; loops++)
      d_sum = thrust::reduce(d_data.begin(), d_data.end(), initVal, fcn1);
    endTime=omp_get_wtime();
    cout << "Thrust  ";
  }
    
  cout << "Ave time for transform reduce " << (endTime-startTime)/NTEST << endl;
  cout << (sizeof(T)*nData/1e9) << " GB " << endl;
  cout << "d_sum   " << d_sum << endl;

  cudaFree(d_partialVals);

#ifdef DO_CHECK
  T testVal = thrust::reduce(d_data.begin(), d_data.end(), initVal, fcn1);
  cout << "testVal " << testVal << endl;
  if(testVal != (d_sum)) {cout << "ERROR " << endl;}
#endif
}

int main(int argc, char* argv[])
{
  if(argc < 3) {
    cerr << "Use: nData(K) op(0:no prefetch, 1:prefetch, 2:thrust)" << endl;
    return(1);
  }
  int nData=(atof(argv[1])*1000000);
  int op=atoi(argv[2]);

  doTest<RUNTYPE>(nData, op);
  return 0;
}
