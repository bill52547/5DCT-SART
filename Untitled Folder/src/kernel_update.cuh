#ifndef _KERNEL_UPDATE_CUH
#define _KERNEL_UPDATE_CUH
#include "universal.cuh"
__global__ void kernel_update(float *img1, float *img, int nx, int ny, int nz, float lambda);
#endif // _KERNEL_UPDATE_CUH