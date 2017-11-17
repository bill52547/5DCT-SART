#ifndef _KERNEL_INITIAL_CUH
#define _KERNEL_INITIAL_CUH
#include "universal.cuh"
__global__ void kernel_initial(float *img, int nx, int ny, int nz, float value);
#endif // _KERNEL_INITIAL_CUH