// live stream kernel (Rob Farber)
// Simple kernel to modify vertex positions in sine wave pattern
__global__ void kernelWave(float4* pos, uchar4 *colorPos,
			   unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

    // write output vertex
    pos[y*width+x] = make_float4(u, w, v, 1.0f);
}

__global__ void kernelFlat(float4* pos, uchar4 *colorPos,
			   unsigned int width, unsigned int height)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  
  // calculate uv coordinates
  float u = x / (float) width;
  float v = y / (float) height;
  u = u*2.0f - 1.0f;
  v = v*2.0f - 1.0f;

  // write output vertex
  pos[y*width+x] = make_float4(u, 1.f, v, 1.0f);
}

__global__ void kernelSkin(float4* pos, uchar4 *colorPos,
			   unsigned int width, unsigned int height,
			   int lowPureG, int highPureG,
			   int lowPureR, int highPureR)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  int r = colorPos[y*width+x].x;
  int g = colorPos[y*width+x].y;
  int b = colorPos[y*width+x].z;
  int pureR = 255*( ((float)r)/(r+g+b));
  int pureG = 255*( ((float)g)/(r+g+b));
  if( !( (pureG > lowPureG) && (pureG < highPureG)
	 && (pureR > lowPureR) && (pureR < highPureR) ) )
    colorPos[y*width+x] = make_uchar4(0,0,0,0);
}

__device__ __forceinline__ unsigned char gray(const uchar4 &pix)
{
  // convert to 8-bit grayscale
  return( .3f * pix.x + 0.59f * pix.y + 0.11f * pix.z);
}
__global__ void kernelSobel(float4 *pos, uchar4 *colorPos, uchar4 *newPix,
			    unsigned int width, unsigned int height)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  
  const int sobelv[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1}};
  const int sobelh[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1}};

  int sumh=0, sumv=0;
  if( (x > 1) && x < (width-1) && (y > 1) && y < (height-1)) {
    for(int l= -1; l < 2; l++) {
      for(int k= -1; k < 2; k++) {
	register int g = gray(colorPos[(y+k)*width+x+l]);
	sumh += sobelh[k+1][l+1] * g;
	sumv += sobelv[k+1][l+1] * g;
      }
    }
    unsigned char p = abs(sumh/8)+ abs(sumv/8); // reuse sumh
    newPix[y*width+x] = make_uchar4(0,p,p,p);
  } else {
    newPix[y*width+x] = make_uchar4(0,0,0,0);
  }
}

extern int PureR[2], PureG[2], doWave, doSkin,doSobel;
// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(float4* pos, uchar4* colorPos,
			      unsigned int mesh_width, 
			      unsigned int mesh_height,
			      float time)
{
  // execute the kernel
  dim3 block(8, 8, 1);
  dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
  if(doWave) 
    kernelWave<<< grid, block>>>(pos, colorPos, mesh_width, mesh_height, time);
  else
    kernelFlat<<< grid, block>>>(pos, colorPos, mesh_width, mesh_height);

  if(doSkin)
    kernelSkin<<< grid, block>>>(pos, colorPos, mesh_width, mesh_height, 
				 PureG[0], PureG[1], 
				 PureR[0], PureR[1]);
  if(doSobel) {
    static uchar4 *newPix=NULL;
    if(!newPix)
      cudaMalloc(&newPix, sizeof(uchar4)*mesh_width*mesh_height);

    kernelSobel<<< grid, block>>>(pos, colorPos, newPix,
				  mesh_width, mesh_height);
    cudaMemcpy(colorPos, newPix, sizeof(uchar4)*mesh_width*mesh_height, 
	       cudaMemcpyDeviceToDevice);
  }
}
