//simpleVBO (Rob Farber)
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

extern float animTime;

////////////////////////////////////////////////////////////////////////////////
// VBO specific code
#include <cutil_inline.h>

#ifndef MESH_WIDTH
#define MESH_WIDTH 640
#endif
#ifndef MESH_HEIGHT
#define MESH_HEIGHT 480
#endif
// constants
const unsigned int mesh_width = MESH_WIDTH;
const unsigned int mesh_height = MESH_HEIGHT;
const unsigned int RestartIndex = 0xffffffff;

typedef struct {
  GLuint vbo;
  GLuint typeSize;
  struct cudaGraphicsResource *cudaResource;
} mappedBuffer_t;

extern "C" 
void launch_kernel(float4* pos, uchar4* posColor,
		   unsigned int mesh_width, unsigned int mesh_height, 
		   float time);

// vbo variables
mappedBuffer_t vertexVBO = {NULL, sizeof(float4), NULL};
mappedBuffer_t colorPBO =  {NULL, sizeof(uchar4), NULL};
GLuint* qIndices=NULL; // index values for primitive restart
int qIndexSize=0;
uchar4 *liveData;

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createBO(mappedBuffer_t* mbuf)
{
  // create buffer object
  glGenBuffers(1, &(mbuf->vbo) );
  glBindBuffer(GL_ARRAY_BUFFER, mbuf->vbo);
  
  // initialize buffer object
  unsigned int size = mesh_width * mesh_height * mbuf->typeSize;
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  cudaGraphicsGLRegisterBuffer( &(mbuf->cudaResource), mbuf->vbo,
				cudaGraphicsMapFlagsNone );
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteBO(mappedBuffer_t* mbuf)
{
  glBindBuffer(1, mbuf->vbo );
  glDeleteBuffers(1, &(mbuf->vbo) );
  
  cudaGraphicsUnregisterResource( mbuf->cudaResource );
  mbuf->cudaResource = NULL;
  mbuf->vbo = NULL;
}

void cleanupCuda()
{
  if(qIndices) free(qIndices);
  deleteBO(&vertexVBO);
  deleteBO(&colorPBO);
  if(liveData) free(liveData);
}

/////////////////////////
// Add the tcp info needed for the live stream
//////////////////////////////////////////////////////////////////////////////
#define TCP_PORT 32000
typedef struct {
   unsigned char r;
   unsigned char g;
   unsigned char b;
} rgb24_t;

uchar4* argbData;
int argbDataSize;
extern int tcpRead(char*, int);
void initTCPserver(int);

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    uchar4 *cptr;
    uint *iptr;
    size_t start;
    cudaGraphicsMapResources( 1, &vertexVBO.cudaResource, NULL );
    cudaGraphicsResourceGetMappedPointer( ( void ** )&dptr, &start, 
					  vertexVBO.cudaResource );
    cudaGraphicsMapResources( 1, &colorPBO.cudaResource, NULL );
    cudaGraphicsResourceGetMappedPointer( ( void ** )&cptr, &start, 
					  colorPBO.cudaResource );

    rgb24_t data[mesh_width*mesh_height];
    if(tcpRead((char*)data, mesh_width*mesh_height*sizeof(rgb24_t) )) {
      // have data
#pragma omp parallel for
      for(int i=0; i < mesh_width*mesh_height; i++) {
	argbData[i].w=0;
	argbData[i].x=data[i].r;
	argbData[i].y=data[i].g;
	argbData[i].z=data[i].b;
      }
      cudaMemcpy(liveData, argbData, argbDataSize, cudaMemcpyHostToDevice);
    }

    //copy the GPU-side buffer to the colorPBO
    cudaMemcpy(cptr, liveData, argbDataSize, cudaMemcpyDeviceToDevice);

    // execute the kernel
    launch_kernel(dptr, cptr, mesh_width, mesh_height, animTime);

    // unmap buffer object
    cudaGraphicsUnmapResources( 1, &vertexVBO.cudaResource, NULL );
    cudaGraphicsUnmapResources( 1, &colorPBO.cudaResource, NULL );
}

void initCuda(int argc, char** argv)
{
  // First initialize OpenGL context, so we can properly set the GL
  // for CUDA.  NVIDIA notes this is necessary in order to achieve
  // optimal performance with OpenGL/CUDA interop.  use command-line
  // specified CUDA device, otherwise use device with highest Gflops/s
  if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
    cutilGLDeviceInit(argc, argv);
  } else {
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
  }
  
  createBO(&vertexVBO);
  createBO(&colorPBO);

  // allocate and assign trianglefan indicies 
  qIndexSize = 5*(mesh_height-1)*(mesh_width-1);
  qIndices = (GLuint *) malloc(qIndexSize*sizeof(GLint));
  int index=0;
  for(int i=1; i < mesh_height; i++) {
    for(int j=1; j < mesh_width; j++) {
      qIndices[index++] = (i)*mesh_width + j; 
      qIndices[index++] = (i)*mesh_width + j-1; 
      qIndices[index++] = (i-1)*mesh_width + j-1; 
      qIndices[index++] = (i-1)*mesh_width + j; 
      qIndices[index++] = RestartIndex;
    }
  }

  cudaMalloc((void**)&liveData,sizeof(uchar4)*mesh_width*mesh_height);

  // make certain the VBO gets cleaned up on program exit
  atexit(cleanupCuda);

  // setup TCP context for live stream
  argbDataSize = sizeof(uchar4)*mesh_width*mesh_height;
  cudaHostAlloc((void**) &argbData, argbDataSize, cudaHostAllocPortable );
  initTCPserver(TCP_PORT);

  runCuda();
}

void renderCuda(int drawMode)
{
  glBindBuffer(GL_ARRAY_BUFFER, vertexVBO.vbo);
  glVertexPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);
  
  glBindBuffer(GL_ARRAY_BUFFER, colorPBO.vbo);
  glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
  glEnableClientState(GL_COLOR_ARRAY);

  switch(drawMode) {
  case GL_LINE_STRIP:
    for(int i=0 ; i < mesh_width*mesh_height; i+= mesh_width)
      glDrawArrays(GL_LINE_STRIP, i, mesh_width);
    break;
  case GL_TRIANGLE_FAN: {
    glPrimitiveRestartIndexNV(RestartIndex);
    glEnableClientState(GL_PRIMITIVE_RESTART_NV);
    glDrawElements(GL_TRIANGLE_FAN, qIndexSize, GL_UNSIGNED_INT, qIndices);
    glDisableClientState(GL_PRIMITIVE_RESTART_NV);
  } break;
  default:
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    break;
  }

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
}

