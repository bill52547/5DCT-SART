#ifndef _KERNEL_FORWARDDVF_CUH
#define _KERNEL_FORWARDDVF_CUH
#include "universal.cuh"
__global__ void kernel_forwardDVF(float *mx, float *my, float *mz, cudaTextureObject_t alpha_x, cudaTextureObject_t alpha_y, cudaTextureObject_t alpha_z, cudaTextureObject_t beta_x, cudaTextureObject_t beta_y, cudaTextureObject_t beta_z, float volume, float flow, int nx, int ny, int nz);

#endif // _KERNEL_FORWARDDVF_CUH