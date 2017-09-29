#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <math.h>

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(x) ((x) > 0 ? (x) : -(x))
#define PI 3.141592653589793
// Set thread block size
#define BLOCKWIDTH 16
#define BLOCKHEIGHT 16 
#define BLOCKDEPTH 4
__global__ void kernel_tex_proj(float *y, cudaTextureObject_t x, float *phi, float SD, float SO, float scale, float dz, int nx, int ny, int nz, int nv, int na, float ai, float da, int nb, float bi, float db);