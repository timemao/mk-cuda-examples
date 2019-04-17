#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "mpi.h"
const static int nGPUperNode=2;

// GPU based autoencoder PCA code

using namespace std;

#include "nelmin.h"

// Define the sigmoidal function
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
  thrust::device_vector<Real> d_data;
  thrust::device_vector<Real> d_param;

 public:
#include "CalcError.h"

  ObjFunc() { nExamples = 0; objFuncCallCount=0; objFuncCallTime=0.;}
  double aveObjFuncWallTime() { return(objFuncCallTime/objFuncCallCount); }
  double totalObjFuncWallTime() { return(objFuncCallTime); }
  int get_nExamples() {return(nExamples);}

  void setExamples(thrust::host_vector<Real>& _h_data) {
    nExamples = _h_data.size()/exLen;

    // copy data to the device
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    cudaSetDevice(rank%nGPUperNode);

    d_data = thrust::device_vector<Real>(nExamples*exLen);
    thrust::copy(_h_data.begin(), _h_data.end(), d_data.begin());
    d_param = thrust::device_vector<Real>(nParam);
  }

  Real objFunc(Real *p)
  {
    int rank,op;
    Real sum=0.;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    cudaSetDevice(rank%nGPUperNode);

    if(nExamples == 0)  {
      cerr << "data not set " <<  endl; exit(1);
    }

    CalcError getError(thrust::raw_pointer_cast(&d_data[0]),
			       thrust::raw_pointer_cast(&d_param[0]),
			       nInput, exLen);

    if(rank > 0) { // slave objective function
      Real *param;
      cudaHostAlloc(&param, sizeof(Real)*nParam,cudaHostAllocPortable); 
      for(;;) { // loop until the master says I am done.
	MPI_Bcast(&op, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(op==0) {
	  cudaFreeHost(param);
	  return(0);
	}
	if(sizeof(Real) == sizeof(float))
	  MPI_Bcast(&param[0], nParam, MPI_FLOAT, 0, MPI_COMM_WORLD);
	else
	  MPI_Bcast(&param[0], nParam, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	thrust::copy(param, param+nParam, d_param.begin());
	Real mySum = thrust::transform_reduce(
			thrust::counting_iterator<unsigned int>(0),
			thrust::counting_iterator<unsigned int>(nExamples),
			getError,
			(Real) 0.,
			thrust::plus<Real>());
	if(sizeof(Real) == sizeof(float))
	  MPI_Reduce(&mySum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	else
	  MPI_Reduce(&mySum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      }
    } else { // master process
      double startTime=omp_get_wtime();
      
      op=1;
      MPI_Bcast(&op, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if(sizeof(Real) == sizeof(float))
	MPI_Bcast(&p[0], nParam, MPI_FLOAT, 0, MPI_COMM_WORLD);
      else
	MPI_Bcast(&p[0], nParam, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      thrust::copy(p, p+nParam, d_param.begin());
      
      Real mySum = thrust::transform_reduce(
			thrust::counting_iterator<unsigned int>(0),
			thrust::counting_iterator<unsigned int>(nExamples),
			getError,
			(Real) 0.,
			thrust::plus<Real>());

      if(sizeof(Real) == sizeof(float))
	MPI_Reduce(&mySum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      else
	MPI_Reduce(&mySum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      objFuncCallTime += (omp_get_wtime() - startTime);
      objFuncCallCount++;
    }
    return(sum);
  }
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
  return 2*(rand()/((float)RAND_MAX)) -1.;
}

template <typename Real, int nInput>
void testNN( const Real *p, const Real *in, Real *out) 
{
  register int index=0;
  
  register Real h2_0 = p[index++]; // bottleneck neuron
  {
    register Real h1_0 = p[index++];
    register Real h1_1 = p[index++];
    register Real h1_2 = p[index++];
    register Real h1_3 = p[index++];
    for(int i=0; i < nInput; i++) {
      register Real input=in[i];
      h1_0 += input * p[index++]; h1_1 += input * p[index++];
      h1_2 += input * p[index++]; h1_3 += input * p[index++];
    }
    h1_0 = G(h1_0); h1_1 = G(h1_1);
    h1_2 = G(h1_2); h1_3 = G(h1_3);
    
    h2_0 += p[index++] * h1_0; h2_0 += p[index++] * h1_1;
    h2_0 += p[index++] * h1_2; h2_0 += p[index++] * h1_3;
  }
  
  register Real h3_0 = p[index++];
  register Real h3_1 = p[index++];
  register Real h3_2 = p[index++];
  register Real h3_3 = p[index++];
  h3_0 += p[index++] * h2_0; h3_1 += p[index++] * h2_0;
  h3_2 += p[index++] * h2_0; h3_3 += p[index++] * h2_0;
  h3_0 = G(h3_0); h3_1 = G(h3_1);
  h3_2 = G(h3_2); h3_3 = G(h3_3);
  
  for(int i=0; i < nInput; i++) {
    register Real o = p[index++];
    o += h3_0 * p[index++]; o += h3_1 * p[index++];
    o += h3_2 * p[index++]; o += h3_3 * p[index++];
    out[i]=o;
  }
}
  
template <typename Real>
void genData(thrust::host_vector<Real> &h_data, int nVec, Real xVar)
{
  Real xMax = 1.1; Real xMin = -xMax;
  Real xRange = (xMax - xMin);
  for(int i=0; i < nVec; i++) {
    Real t = xRange * f_rand();
    Real z1 = t +  xVar * f_rand();
    Real z2 = t*t*t +  xVar * f_rand();
    h_data.push_back( z1 ); 
    h_data.push_back( z2 );
  }
}

#include <fstream>
template <typename Real>
void trainTest(char* filename, int rank, int numtasks)
{
  ObjFunc<Real> testObj;
  const int nParam = testObj.nParam;
  cout << "nParam " << nParam << endl;

  // read the test data
  ifstream inFile (filename, ios::in | ios::binary);
  // position 0 bytes from end
  inFile.seekg(0, ios::end);
  // determine the file size in bytes
  ios::pos_type size = inFile.tellg();  
  // allocate number of Real values for this task
  //    (assumes a multiple of numtasks)
  int nExamplesPerGPU = size/sizeof(Real)/testObj.exLen/numtasks;
  thrust::host_vector<Real> h_data(nExamplesPerGPU*testObj.exLen);
  // seek to the byte location in the file
  inFile.seekg(rank*h_data.size()*sizeof(Real), ios::beg);
  // read bytes from the file into h_data
  inFile.read((char*)&h_data[0], h_data.size()*sizeof(Real));
  // close the file
  inFile.close();
  
  testObj.setExamples(h_data);
  int nExamples = testObj.get_nExamples();

  if(rank > 0) {
    testObj.objFunc( NULL );
    return;
  } 

  cout << "GB data " << (h_data.size()*sizeof(Real)/1e9) << endl;
  
  // set the Nelder-Mead starting conditions
  int icount, ifault, numres;
  vector<Real> start(nParam);
  vector<Real> step(nParam,1.);
  vector<Real> xmin(nParam);

  srand(0);
  for(int i=0; i < start.size(); i++) start[i] = 0.2 * f_rand();

  Real ynewlo = testObj.objFunc( &start[0] );
  Real reqmin = 1.0E-18;
  int konvge = 10;
  int kcount = 500000; // set large for high-precision

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
  
  cout << " -- Generate scatter plot  -- " << endl;
  int index=0, nTest=100;
  cout << "original input calculated" << endl;
  thrust::host_vector<Real> h_test;
  thrust::host_vector<Real> h_in(testObj.nInput);
  thrust::host_vector<Real> h_out(testObj.nOutput);
  genData<Real>(h_test, nTest, 0.0); // note: no variance for the test
  for(int i=0; i< nTest; i++) {
    h_in[0] = h_test[index++];
    h_in[1] = h_test[index++];
    
    testNN<Real,2>(&xmin[0],&h_in[0],&h_out[0]);
    cout << h_data[testObj.nInput*i] << "," << h_data[testObj.nInput*i+1] << " "
	 << h_in[0] << "," << h_in[1] << " "
	 << h_out[0] << "," << h_out[1] << endl;
  }
  int op=0; // shutdown slave processes

  MPI_Bcast(&op, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

#include <stdio.h>

int main(int argc, char *argv[])
{
  int  numtasks, rank, ret; 
  
  if(argc < 2) {
    fprintf(stderr,"Use: filename\n");
    exit(1);
  }
  ret = MPI_Init(&argc,&argv);
  if (ret != MPI_SUCCESS) {
    printf ("Error in MPI_Init()!\n");
    MPI_Abort(MPI_COMM_WORLD, ret);
  }
  
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  printf ("Number of tasks= %d My rank= %d\n", numtasks,rank);
  
  /*******  do some work *******/
#ifdef USE_DBL
  trainTest<double> ( argv[1], rank, numtasks );
#else
  trainTest<float> ( argv[1], rank, numtasks);
#endif
  
  MPI_Finalize();
  return 0;
}
