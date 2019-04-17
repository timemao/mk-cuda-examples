#include <iostream>
#include <fstream>
#include <stdlib.h>
using namespace std;

// get a uniform random number between -1 and 1 
inline float f_rand() {
  return 2*(rand()/((float)RAND_MAX)) -1.;
}
template <typename Real>
void genData(ofstream &outFile, int nVec, Real xVar)
{
  Real xMax = 1.1; Real xMin = -xMax;
  Real xRange = (xMax - xMin);
  for(int i=0; i < nVec; i++) {
    Real t = xRange * f_rand();
    Real z1 = t +  xVar * f_rand();
    Real z2 = t*t*t +  xVar * f_rand();
    outFile.write((char*) &z1, sizeof(Real));
    outFile.write((char*) &z2, sizeof(Real));
  }
}

int main(int argc, char *argv[])
{
  if(argc < 3) {
    fprintf(stderr,"Use: filename nVec\n");
    exit(1);
  }
  ofstream outFile (argv[1], ios::out | ios::binary);
  int nVec = atoi(argv[2]);
#ifdef USE_DBL
  genData<double>(outFile, nVec, 0.1);
#else
  genData<float>(outFile, nVec, 0.1);
#endif
  outFile.close();
  return 0;
}
