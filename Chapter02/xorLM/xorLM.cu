#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// GPU based xor test code
// define -D USE_HOST for a host-based version

using namespace std;

#include "levmar.h"

// Note: use -use_fast_math for best performance
__device__ __host__
inline float G(float x) { return( tanhf(x) ) ;} 
__device__ __host__
inline double G(double x) { return( tanh(x) );} 

// This is a convience class to hold all the examples and 
// archtecture information. Most is boilerplate. CalcError
// is where all the work happens.
template<typename Real>
class ObjFunc {
 private:
  double objFuncCallTime;
  unsigned int objFuncCallCount;
 protected:
  int nExamples;
#ifdef USE_HOST
  thrust::host_vector<Real> h_data;
#else
  thrust::device_vector<Real> d_data;
#endif
  thrust::device_vector<Real> d_param;
  thrust::device_vector<Real> d_x;

 public:
  // The CalcError functor goes here
#include "CalcError.h"
#ifdef FOO
  static const int nInput = 2;
  static const int nH1 = 1;
  static const int nOutput = 1;
  static const int nParam = 
    (nOutput+nH1) // Neuron Offsets
    + (nInput*nH1) // connections from I to H1
    + (nH1*nOutput) // connections from H1 to O
    + (nInput*nOutput); // connections from I to O
  static const int exLen = nInput + nOutput;

  struct CalcError {
    const Real* examples;
    const Real* p;
    const int nInput;
    const int exLen;
    Real* x;
    
// Note this returns the error (not error^2)
  CalcError( const Real* _examples, const  Real* _p,
	     const int _nInput, const int _exLen, Real* _x)
  : examples(_examples), p(_p), nInput(_nInput), exLen(_exLen), x(_x) {};
    
    __device__ __host__
    void operator()(const unsigned int& tid)
    {
      const register Real* in = &examples[tid * exLen];
      register int index=0;
      register Real h1 = p[index++];
      register Real o = p[index++];
      
      h1 += in[0] * p[index++];
      h1 += in[1] * p[index++];
      h1 = tanhf(h1);
      
      o += in[0] * p[index++];
      o += in[1] * p[index++];
      o += h1 * p[index++];
      
      // calculate the error
      x[tid] = o - in[nInput];
    }
  };
#endif

  // Boilerplate constructor and helper classes
  ObjFunc() { nExamples = 0; objFuncCallCount=0; objFuncCallTime=0.;}
  
  double aveObjFuncWallTime() { return(objFuncCallTime/objFuncCallCount); }
  double totalObjFuncWallTime() { return(objFuncCallTime); }
  int get_nExamples() {return(nExamples);}

  void setExamples(thrust::host_vector<Real>& _h_data) {
#ifdef USE_HOST
    h_data = _h_data;
#else
    d_data = _h_data;
#endif
    nExamples = _h_data.size()/exLen;
    d_param = thrust::device_vector<Real>(nParam);
    d_x = thrust::device_vector<Real>(nOutput * nExamples);
  }

#ifdef USE_HOST
  void mFunc(Real* p, Real* x, int m, int n, void* data)
  {
    if(nExamples == 0)  { cerr << "data not set" << endl; exit(1); }

    double startTime=omp_get_wtime();

    CalcError getError(&h_data[0], p, nInput, exLen, x);

    // get the error vector using OpenMP
#pragma omp parallel for
    for(int i=0; i < nExamples; ++i) getError(i);

    objFuncCallTime += (omp_get_wtime() - startTime);
    objFuncCallCount++;
  }
#else
  void mFunc(Real* p, Real* x, int m, int n, void* data)
  {
    if(nExamples == 0)  { cerr << "data not set " <<  endl; exit(1); }
    
    double startTime=omp_get_wtime();
    thrust::copy(p, p+nParam, d_param.begin());
    // Note: d_x is initialized by getError on each call
    
    CalcError getError(thrust::raw_pointer_cast(&d_data[0]),
		       thrust::raw_pointer_cast(&d_param[0]),
		       nInput, exLen, 
		       thrust::raw_pointer_cast(&d_x[0]) );
    
    thrust::for_each( thrust::counting_iterator<unsigned int>(0),
		      thrust::counting_iterator<unsigned int>(nExamples),
		      getError);
    thrust::copy(d_x.begin(), d_x.end(), x);

    objFuncCallTime += (omp_get_wtime() - startTime);
    objFuncCallCount++;
  }
#endif
};

// Wrapper so the objective function can be called 
// as a pointer to function for C-style libraries.
// Note: polymorphism allows easy use of 
// either float or double types.
void* objFunc_object=NULL;
void lm_func(float* p, float* x, int m, int n, void* data)
{
  if(objFunc_object)
    ((ObjFunc<float>*) objFunc_object)->mFunc(p,x,m,n,data);
}
void lm_func(double* p, double* x, int m, int n, void* data)
{
  if(objFunc_object) 
    ((ObjFunc<double>*) objFunc_object)->mFunc(p,x,m,n,data);
}

 //wrapper around the levmar single and double precision calls
inline int levmar_dif( void (*func)(float *, float *, int, int, void *),
		       float *p, float *x, int m, int n, int itmax,
		       float *opts, float *info, float *work, float *covar,
		       void* data)
{
  return slevmar_dif(func, p, x, m, n, itmax, opts, info, work, covar, data);
}
inline int levmar_dif( void (*func)(double *, double *, int, int, void *),
		       double *p, double *x, int m, int n, int itmax,
		       double *opts, double *info, double *work, double *covar,
		       void* data)
{
  return dlevmar_dif(func, p, x, m, n, itmax, opts, info, work, covar, data);
}

// get a uniform random number between -1 and 1 
inline float f_rand() {
  return 2*(rand()/((float)RAND_MAX)) -1.;
}

template <typename Real, int nInput>
void testNN( const Real *p, const Real *in, Real *out) 
{
  int index=0;
  
  Real h1 = p[index++];
  Real o = p[index++];
  
  h1 += in[0] * p[index++];
  h1 += in[1] * p[index++];
  h1 = G(h1);
  
  o += in[0] * p[index++];
  o += in[1] * p[index++];
  o += h1 * p[index++];
  
  out[0]=o;
}
  
template <typename Real>
void genData(thrust::host_vector<Real> &h_data, int nVec, Real xVar)
{
  // Initialize the data via replication of the XOR truth table
  Real dat[] = {
    0.1, 0.1, 0.1,
    0.1, 0.9, 0.9,
    0.9, 0.1, 0.9,
    0.9, 0.9, 0.1};
  
  for(int i=0; i < nVec; i++) 
    for(int j=0; j < 12; j++) h_data.push_back(dat[j] + xVar * f_rand() );
}

template <typename Real>
void testTraining()
{
  ObjFunc<Real> testObj;
  const int nParam = testObj.nParam;
  cout << "nParam " << nParam << endl;

  // generate the test data
  const int nVec=1000 * 1000 * 10;
  thrust::host_vector<Real> h_data;
  genData<Real>(h_data, nVec, 0.01);
  testObj.setExamples(h_data);
  cout << "GB data " << (h_data.size()*sizeof(Real)/1e9) << endl;
  int nExamples = testObj.get_nExamples();
  
  thrust::host_vector<Real> h_p(nParam);
  thrust::host_vector<Real> h_x(nExamples*testObj.nOutput,0.0);

  // pick a some random starting parameters
  srand(0);
  for(int i=0; i < h_p.size(); i++) h_p[i] = 0.25 * f_rand();

  objFunc_object = &testObj;

  // specify the Levmar runtime options
  Real opts[] = {LM_INIT_MU, 1e-15, 1e-15, LM_DIFF_DELTA};
  Real info[LM_INFO_SZ];

  double optStartTime=omp_get_wtime();
  // call the method that approximates the Jacobian
  int ret=levmar_dif(lm_func, &h_p[0], &h_x[0], nParam, nExamples,
		     1000, opts, info, NULL, NULL, NULL);
  double optTime=omp_get_wtime()-optStartTime;
  
  cout << "Levenberg-Marquardt returned " << ret << " in " << info[5]
       << " iter "  << endl;
  switch((int)info[6]) {
  case 1: cout << "- stopped by small gradient J^T e" << endl; break;
  case 2: cout << "- stopped by small Dp" << endl; break;
  case 3: cout << "- stopped by itmax" << endl; break;
  case 4: cout << 
      "- singular matrix. Restart from current p with increased mu" << endl;
    break;
  case 5: cout << 
      "- no further error reduction is possible. Restart with increased mu"
	       << endl; break;
  case 6: cout << "- stopped by small ||e||_2" << endl; break;
  case 7: cout << "- stopped by NaN or Inf func values ... user error"
	       << endl; break;
  default: cout << "unknown reason" << endl; break;
  }
  
  cout << "Average wall time for ObjFunc " 
       << testObj.aveObjFuncWallTime() << endl;
  cout << "Total wall time in optimization method " << optTime << endl;
  cout << "Percent time in objective function " << 
    (100.*(testObj.totalObjFuncWallTime()/optTime)) << endl;
  
  int index=0, nTest=4;
  cout << "pred known" << endl;
  thrust::host_vector<Real> h_test;
  thrust::host_vector<Real> h_in(testObj.nInput);
  thrust::host_vector<Real> h_out(testObj.nOutput);
  genData<Real>(h_test, nTest, 0.0); // note: no variance for the test
  for(int i=0; i< nTest; i++) {
    h_in[0] = h_test[index++];
    h_in[1] = h_test[index++];
    
    testNN<Real,2>(&h_p[0],&h_in[0],&h_out[0]);
    cout << setprecision(1) << setw(4) 
	 << h_out[0] << " " 
	 << h_test[index] << endl;
    index++;
  }
}

int main ( )
{
#ifdef USE_DBL
  testTraining<double> ( );
#else
  testTraining<float> ( );
#endif
  return 0;
}

