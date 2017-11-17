#ifndef _KERNEL_DIVISION_CUH
#define _KERNEL_DIVISION_CUH
#include "universal.cuh"
__global__ void kernel_division(float *img1, float *img, int nx, int ny, int nz);

#endif // _KERNEL_DIVISION_CUH
