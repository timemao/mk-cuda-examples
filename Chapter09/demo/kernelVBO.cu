// sinusoidal kernel (Rob Farber)
// Simple kernel to modify vertex positions in sine wave pattern
__global__ void kernel(float4* pos, uchar4 *colorPos,
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
    colorPos[y*width+x].w = 0;
    colorPos[y*width+x].x = 255.f *0.5*(1.f+sinf(w+x));
    colorPos[y*width+x].y = 255.f *0.5*(1.f+sinf(x)*cosf(y));
    colorPos[y*width+x].z = 255.f *0.5*(1.f+sinf(w+time/10.f));
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(float4* pos, uchar4* colorPos,
			      unsigned int mesh_width, unsigned int mesh_height,
			      float time)
{
  // execute the kernel
  dim3 block(8, 8, 1);
  dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
  kernel<<< grid, block>>>(pos, colorPos, mesh_width, mesh_height, time);
}
