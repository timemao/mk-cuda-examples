#include <iostream>
#include <cassert>
using namespace std;
#include <cuda.h>
#define CUFFT_FORWARD -1
#define CUFFT_INVERSE  1

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include <cufft.h>

#ifndef REAL
#define REAL float
#endif

template <typename Real>
class DistFFT3D {
 protected:
  int nGPU;
  cudaStream_t *streams;
  int nPerCall;
  vector<cufftHandle> fftPlanMany;
  int dim[3];
  Real *h_data;
  long h_data_elements;
  long nfft, n2ft3d, h_memsize, nelements;
  long totalFFT;
  vector<Real *> d_data;
  long bytesPerGPU;

 public:
  DistFFT3D(int _nGPU, Real* _h_data, long _h_data_elements,
	    int *_dim, int _nPerCall, cudaStream_t *_streams) {
    nGPU= _nGPU;
    h_data = _h_data;
    h_data_elements = _h_data_elements;
    dim[0] = _dim[0]; dim[1] = _dim[1]; dim[2] = _dim[2];
    nfft = dim[0]*dim[1]*dim[2];
    n2ft3d = 2*dim[0]*dim[1]*dim[2];
    totalFFT = h_data_elements/n2ft3d;

    set_NperCall(_nPerCall);
    bytesPerGPU = nPerCall*n2ft3d*sizeof(Real);
    h_memsize = h_data_elements*sizeof(Real);
    assert( (totalFFT/nPerCall*bytesPerGPU) == h_memsize);

    streams = _streams;
    fftPlanMany = vector<cufftHandle>(nGPU);

    initFFTs();

    for(int i=0; i<nGPU; i++) {
      Real* tmp;
      cudaSetDevice(i);
      if(cudaMalloc(&tmp,bytesPerGPU)) {
	cerr << "Cannot allocate space on device!" << endl;
	exit(1);
      }
      d_data.push_back(tmp);
    }

  }
  void set_NperCall(int n) {
    cerr << "Setting nPerCall " << n << endl;
    nPerCall = n;
    if( (nGPU * nPerCall) > totalFFT) {
      cerr << "Too many nPerCall specified! max " << (totalFFT/nGPU) << endl;
      exit(1);
    }
  }
  ~DistFFT3D() {
    for(int i=0; i < nGPU; i++) {
      cudaSetDevice(i);
      cudaFree(d_data[i]);
    }
  }

  void inline initFFTs() 
  {
    if((nPerCall*nGPU) > totalFFT) {
      cerr << "nPerCall must be a multiple of totalFFT" << endl;
      exit(1);
    }
    // Create a batched 3D plan
    for(int sid=0; sid < nGPU; sid++) {
      cudaSetDevice(sid);
      if(sizeof(Real) == sizeof(float) ) {
	cufftPlanMany(&fftPlanMany[sid], 3, dim, NULL, 1, 0, NULL, 1, 0,
		      CUFFT_C2C,nPerCall);
      } else {
	cufftPlanMany(&fftPlanMany[sid], 3, dim, NULL, 1, 0, NULL, 1, 0, 
		      CUFFT_Z2Z,nPerCall);
      }
      if(cufftSetStream(fftPlanMany[sid],streams[sid])) {
	cerr << "cufftSetStream failed!" << endl;
      }
    }
    cudaSetDevice(0);
  }
  
  inline void _FFTerror(int ret) {
    switch(ret) {
    case CUFFT_SETUP_FAILED: cerr << "SETUP_FAILED" << endl; break;
    case CUFFT_INVALID_PLAN: cerr << "INVALID_PLAN" << endl; break;
    case CUFFT_INVALID_VALUE: cerr << "INVALID_VALUE" << endl; break;
    case CUFFT_EXEC_FAILED: cerr << "EXEC_FAILED" << endl; break;
    default: cerr << "UNKNOWN ret code " << ret << endl;
    }
  }

//template specialization to handle different data types (float,double) 
inline void cinverseFFT_(cufftHandle myFFTplan, float* A, float* B ) {
  int ret=cufftExecC2C(myFFTplan, (cufftComplex*)A,
		       (cufftComplex*) B, CUFFT_INVERSE);
  if(ret != CUFFT_SUCCESS) {
    cerr << "C2C FFT failed! ret code " << ret << endl;
    _FFTerror(ret); exit(1);
  }
}
 
 inline void cinverseFFT_(cufftHandle myFFTplan, double *A, double *B) {
  int ret = cufftExecZ2Z(myFFTplan, (cufftDoubleComplex*)A,
			 (cufftDoubleComplex*) B, CUFFT_INVERSE);
  
  if(ret != CUFFT_SUCCESS) {
    cerr << "Z2Z FFT failed! ret code " << ret << endl; 
    _FFTerror(ret); exit(1);
  }
 }
 
 inline void cforwardFFT_(cufftHandle myFFTplan, float* A, float* B ) {
   int ret = cufftExecC2C(myFFTplan, (cufftComplex*)A,
			  (cufftComplex*) B, CUFFT_FORWARD);
   
   if(ret != CUFFT_SUCCESS) {
     cerr << "C C2C FFT failed!" << endl; _FFTerror(ret); exit(1);
   }
 }
 
 inline void cforwardFFT_(cufftHandle myFFTplan, double *A, double *B) {
   int ret = cufftExecZ2Z(myFFTplan, (cufftDoubleComplex*)A,
			  (cufftDoubleComplex*) B, CUFFT_FORWARD);
   
   if(ret != CUFFT_SUCCESS) {
     cerr << "Z2Z FFT failed!" << endl; _FFTerror(ret); exit(1);
   }
 }
 
 double showError(Real* h_A1)
 {
   double error=0.;
#pragma parallel for reduction (+ : error)
   for(int i=0; i < h_data_elements; i++) {
     h_data[i] /= (Real)nfft;
     error += abs(h_data[i] - h_A1[i]);
   }
   return error;
 }
 void doit() 
 {
   
   double startTime = omp_get_wtime();
   long h_offset=0;
   for(int i=0; i < totalFFT; i += nGPU*nPerCall) {
     for(int j=0; j < nGPU; j++) {
       cudaSetDevice(j);
       cudaMemcpyAsync(d_data[j], ((char*)h_data)+h_offset,
		       bytesPerGPU, cudaMemcpyDefault,streams[j]);
       cforwardFFT_(fftPlanMany[j],d_data[j], d_data[j]);
       cinverseFFT_(fftPlanMany[j],d_data[j], d_data[j]);
       cudaMemcpyAsync(((char*)h_data)+h_offset, d_data[j],
		       bytesPerGPU, cudaMemcpyDefault,streams[j]);
       h_offset += bytesPerGPU;
     }
   }
   cudaDeviceSynchronize();
   cudaSetDevice(0);
   
   double endTime = omp_get_wtime();

   cout << dim[0] << " " << dim[1] << " " << dim[2] 
	<< " nFFT/s  " << 1./(0.5*(endTime-startTime)/totalFFT)
	<< " average 3D fft time " << (0.5*(endTime-startTime)/totalFFT)
	<< " total " << (endTime-startTime) << endl;
 }
};
 
main(int argc, char *argv[])
{
  if(argc < 6) {
    cerr << "Use nGPU dim[0] dim[1] dim[2] numberFFT [nFFT per call)" << endl;
    exit(1);
  }
  
  int nPerCall = 1;
  int nGPU = atoi(argv[1]);
  int dim[] = { atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};
  int totalFFT=atoi(argv[5]);
  nPerCall = totalFFT;
  if( argc > 6) {
    nPerCall = atoi(argv[6]);
    if(totalFFT % nPerCall != 0) {
      cerr << "nPerCall must be a multiple of totalFFT!" << endl;
      return(1);
    }
  }

  int systemGPUcount;
  cudaGetDeviceCount(&systemGPUcount);
  if(nGPU > systemGPUcount) {
    cerr << "Attempting to use too many GPUs!" << endl;
    return(1);
  }

  cerr << "nGPU = " << nGPU << endl;
  cerr << "dim[0] = " << dim[0] << endl;
  cerr << "dim[1] = " << dim[1] << endl;
  cerr << "dim[2] = " << dim[2] << endl;
  cerr << "totalFFT = " << totalFFT << endl;
  cerr << "sizeof(REAL) is " << sizeof(REAL) << " bytes" << endl;
  
  cudaStream_t streams[nGPU];
  for(int sid=0; sid < nGPU; sid++) {
    cudaSetDevice(sid);
    if(cudaStreamCreate(&streams[sid]) != 0) {
      cerr << "Stream create failed!" << endl;
    }
  }
  cudaSetDevice(0);

  long nfft = dim[0]*dim[1]*dim[2];
  long n2ft3d = 2*dim[0]*dim[1]*dim[2];
  long nelements = n2ft3d*totalFFT;

  REAL *h_A, *h_A1; 
  if(cudaHostAlloc(&h_A, nelements*sizeof(REAL), cudaHostAllocPortable)
     != cudaSuccess) {
    cerr << "cudaHostAlloc failed!|" << endl; exit(1);
  }
  h_A1 = (REAL*) malloc(nelements*sizeof(REAL));
  if(!h_A1) {
    cerr << "malloc failed!" << endl; exit(1);
  }

  // fill the test data
#pragma parallel for
   for(long i=0; i < nelements; i++)  h_A1[i] = h_A[i] = i%n2ft3d;
   
   DistFFT3D<REAL> dfft3d(nGPU, h_A, nelements, dim, nPerCall,  streams);
   dfft3d.doit();
   double error = dfft3d.showError(h_A1);
   cout << "average error per fft " << (error/nfft/totalFFT) << endl;
   cudaFreeHost(h_A1);
}

