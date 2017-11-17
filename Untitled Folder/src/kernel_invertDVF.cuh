#ifndef _KERNEL_INVERTDVF_CUH
#define _KERNEL_INVERTDVF_CUH
#include "universal.cuh"
__global__ void kernel_invertDVF(float *mx2, float *my2, float *mz2, cudaTextureObject_t mx, cudaTextureObject_t my, cudaTextureObject_t mz, int nx, int ny, int nz, int niter);
#endif // _KERNEL_INVERTDVF_CUH