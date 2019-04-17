//Perlin kernel (Rob Farber)
#include <cutil_math.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>

extern float gain, xStart, yStart, zOffset, octaves, lacunarity;
#define Z_PLANE 50.f

__constant__ unsigned char c_perm[256];
__shared__ unsigned char s_perm[256]; // shared memory copy of permuation array
unsigned char* d_perm=NULL; // global memory copy of permutation array
// host version of permutation array
const static unsigned char h_perm[] = {151,160,137,91,90,15,
   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
   190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
   88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
   77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
   102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
   135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
   5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
   223,183,170,213,119,248,152,2,44,154,163, 70,221,153,101,155,167, 43,172,9,
   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
   251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
   49,192,214, 31,181,199,106,157,184,84,204,176,115,121,50,45,127, 4,150,254,
   138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
   };

__device__ inline int perm(int i) { return(s_perm[i&0xff]); }
__device__ inline float fade(float t) { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); }
__device__ inline float lerpP(float t, float a, float b) { return a + t * (b - a); }
__device__ inline float grad(int hash, float x, float y, float z) {
  int h = hash & 15;                      // CONVERT LO 4 BITS OF HASH CODE
  float u = h<8 ? x : y,                 // INTO 12 GRADIENT DIRECTIONS.
    v = h<4 ? y : h==12||h==14 ? x : z;
  return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
}

__device__ float inoise(float x, float y, float z) {
  int X = ((int)floorf(x)) & 255, // FIND UNIT CUBE THAT
    Y = ((int)floorf(y)) & 255,   // CONTAINS POINT.
    Z = ((int)floorf(z)) & 255;
  x -= floorf(x);               // FIND RELATIVE X,Y,Z
  y -= floorf(y);               // OF POINT IN CUBE.
  z -= floorf(z);
  float u = fade(x),            // COMPUTE FADE CURVES
    v = fade(y),                // FOR EACH OF X,Y,Z.
    w = fade(z);
  int A = perm(X)+Y, AA = perm(A)+Z, AB = perm(A+1)+Z, // HASH COORDINATES OF
    B = perm(X+1)+Y, BA = perm(B)+Z, BB = perm(B+1)+Z; // THE 8 CUBE CORNERS,
  
  return lerpP(w, lerpP(v, lerpP(u, grad(perm(AA), x  , y  , z   ), // AND ADD
				 grad(perm(BA), x-1.f, y  , z   )),   // BLENDED
			lerpP(u, grad(perm(AB), x  , y-1.f, z   ),    // RESULTS
			      grad(perm(BB), x-1.f, y-1.f, z   ))),     // FROM  8
	       lerpP(v, lerpP(u, grad(perm(AA+1), x  , y  , z-1.f ),  // CORNERS
			      grad(perm(BA+1), x-1.f, y  , z-1.f )),    // OF CUBE
		     lerpP(u, grad(perm(AB+1), x  , y-1.f, z-1.f ),
			   grad(perm(BB+1), x-1.f, y-1.f, z-1.f ))));
#ifdef ORIG
  return(perm(X));
#endif
 
}

__device__ float fBm(float x, float y, int octaves,
		     float lacunarity = 2.0f, float gain = 0.5f)
{
  float freq = 1.0f, amp = 0.5f;
  float sum = 0.f;  
  for(int i=0; i<octaves; i++) {
    sum += inoise(x*freq, y*freq, Z_PLANE)*amp;
    freq *= lacunarity;
    amp *= gain;
  }
  return sum;
}


__device__ inline uchar4 colorElevation(float texHeight)
{
  uchar4 pos;

  // color textel (r,g,b,a)
       if (texHeight < -1.000f) pos = make_uchar4(000, 000, 128, 255); //deeps
  else if (texHeight < -.2500f) pos = make_uchar4(000, 000, 255, 255); //shallow
  else if (texHeight < 0.0000f) pos = make_uchar4(000, 128, 255, 255); //shore
  else if (texHeight < 0.0125f) pos = make_uchar4(240, 240, 064, 255); //sand
  else if (texHeight < 0.1250f) pos = make_uchar4(032, 160, 000, 255); //grass
  else if (texHeight < 0.3750f) pos = make_uchar4(224, 224, 000, 255); //dirt
  else if (texHeight < 0.7500f) pos = make_uchar4(128, 128, 128, 255); //rock
  else                          pos = make_uchar4(255, 255, 255, 255); //snow

  return(pos);
}

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//Simple kernel fills an array with perlin noise
__global__ void k_perlin(float4* pos, uchar4 *colorPos,
			  unsigned int width, unsigned int height, 
			 float2 start, float2 delta, float gain, float zOffset,
			 unsigned char* d_perm, float octaves, float lacunarity)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float xCur = start.x + ((float) (idx%width)) * delta.x;
  float yCur = start.y + ((float) (idx/width)) * delta.y;

  if(threadIdx.x < 256)
    // Optimization: this causes bank conflicts
    s_perm[threadIdx.x] = d_perm[threadIdx.x];
  // this synchronization can be important if there are more that 256 threads
  __syncthreads();

  
  // Each thread creates one pixel location in the texture (textel)
  if(idx < width*height) {
    float w = fBm(xCur, yCur, octaves, lacunarity, gain) + zOffset;
    
    colorPos[idx] = colorElevation(w);
    float u = ((float) (idx%width))/(float) width;
    float v = ((float) (idx/width))/(float) height;
    u = u*2.f - 1.f;
    v = v*2.f - 1.f;
    w = (w>0.f)?w:0.f; // don't show region underwater
    pos[idx] = make_float4( u, w, v, 1.0f);
  }
}


uchar4 *eColor=NULL;
// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(float4 *pos, uchar4* posColor,
			      unsigned int image_width, unsigned int image_height,
			      float time)
{
  int nThreads=256; // must be equal or larger than 256! (see s_perm)
  int totalThreads = image_height * image_width;
  int nBlocks = totalThreads/nThreads; 
  nBlocks += ((totalThreads%nThreads)>0)?1:0;
  
  float xExtent = 10.f;
  float yExtent = 10.f;
  float xDelta = xExtent/(float)image_width;
  float yDelta = yExtent/(float)image_height;
  
  if(!d_perm) { // for convenience allocate and copy d_perm here
    cudaMalloc((void**) &d_perm,sizeof(h_perm));
    cudaMemcpy(d_perm,h_perm,sizeof(h_perm),cudaMemcpyHostToDevice);
    checkCUDAError("d_perm malloc or copy failed!");
  }

  k_perlin<<< nBlocks, nThreads>>>(pos, posColor, image_width, image_height,
				   make_float2(xStart, yStart),
				   make_float2(xDelta, yDelta),
				   gain, zOffset, d_perm,
				   octaves, lacunarity);
  
  // make certain the kernel has completed 
  cudaThreadSynchronize();
  checkCUDAError("kernel failed!");
}

