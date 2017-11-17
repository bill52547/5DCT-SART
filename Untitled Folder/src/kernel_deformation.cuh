#ifndef _KERNEL_DEFORMATION_CUH
#define _KERNEL_DEFORMATION_CUH
#include "universal.cuh"
__global__ void kernel_deformation(float *img1, cudaTextureObject_t img, float *mx2, float *my2, float *mz2, int nx, int ny, int nz);
#endif // _KERNEL_DEFORMATION_CUH