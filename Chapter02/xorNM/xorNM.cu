#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// define USE_HOST for a host-based version
// define USE_DBL for double-precision
using namespace std;

#include "nelmin.h"
// Define the sigmoidal function
__device__ __host__
inline float G(float x) { return( tanhf(x) ) ;} 
__device__ __host__
inline double G(double x) { return( tanh(x) );} 
// This is a convenience class to hold all the examples and 
// architecture information. Most is boilerplate. CalcError
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

 public:
  // The CalcError functor goes here
#include "CalcError.h"
  
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
  }

#ifdef USE_HOST
  Real objFunc(Real *p) {
    if(nExamples == 0)  { cerr << "data not set" << endl; exit(1); }

    double startTime=omp_get_wtime();

    Real sum = 0.;
    CalcError getError(&h_data[0], p, nInput, exLen);

#pragma omp parallel for reduction(+ : sum)
    for(int i=0; i < nExamples; ++i) {
      Real d = getError(i);
      sum += d;
    }

    objFuncCallTime += (omp_get_wtime() - startTime);
    objFuncCallCount++;
    return(sum);
  }
#else
  Real objFunc(Real *p)
  {
    if(nExamples == 0)  { cerr << "data not set " <<  endl; exit(1); }

    double startTime=omp_get_wtime();

    thrust::copy(p, p+nParam, d_param.begin());

    CalcError getError(thrust::raw_pointer_cast(&d_data[0]),
		       thrust::raw_pointer_cast(&d_param[0]),
		       nInput, exLen);
    Real sum = thrust::transform_reduce(
			thrust::counting_iterator<unsigned int>(0),
			thrust::counting_iterator<unsigned int>(nExamples),
			getError,
			(Real) 0.,
			thrust::plus<Real>());
    objFuncCallTime += (omp_get_wtime() - startTime);
    objFuncCallCount++;
    return(sum);
  }
#endif
};
// Wrapper so the objective function can be called 
// as a pointer to function for C-style libraries.
// Note: polymorphism allows easy use of 
// either float or double types.
void* objFunc_object=NULL;
float func(float* param)
{
  if(objFunc_object)
    return ((ObjFunc<float>*) objFunc_object)->objFunc(param);
  return(0.);
}
double func(double* param)
{
  if(objFunc_object) 
    return ((ObjFunc<double>*) objFunc_object)->objFunc(param);
  return(0.);
}

// get a uniform random number between -1 and 1 
inline float f_rand() {
  return 2.*(rand()/((float)RAND_MAX)) -1.;
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
  int nExamples = testObj.get_nExamples();
  cout << "GB data " << (h_data.size()*sizeof(Real)/1e9) << endl;
  
  // set the Nelder-Mead starting conditions
  int icount, ifault, numres;
  vector<Real> start(nParam);
  vector<Real> step(nParam,1.);
  vector<Real> xmin(nParam);

  srand(0);
  for(int i=0; i < start.size(); i++) start[i] = 0.25 * f_rand();

  Real ynewlo = testObj.objFunc( &start[0] );
  Real reqmin = 1.0E-18;
  int konvge = 10;
  int kcount = 5000;

  objFunc_object = &testObj;
  double optStartTime=omp_get_wtime();
  nelmin<Real> (func, nParam, &start[0], &xmin[0], &ynewlo, reqmin, &step[0],
		  konvge, kcount, &icount, &numres, &ifault );
  double optTime=omp_get_wtime()-optStartTime;

  cout << endl <<"  Return code IFAULT = " << ifault << endl << endl;
  cout << "  Estimate of minimizing value X*:" << endl << endl;
  cout << "  F(X*) = " << ynewlo << endl;
  cout << "  Number of iterations = " << icount << endl;
  cout << "  Number of restarts =   " << numres << endl << endl;

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
    
    testNN<Real,2>(&xmin[0],&h_in[0],&h_out[0]);
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
