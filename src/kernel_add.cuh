#ifndef _KERNEL_ADD_CUH
#define _KERNEL_ADD_CUH
#include "universal.cuh"
__global__ void kernel_add(float *proj1, float *proj, int iv, int na, int nb, float weight);
__global__ void kernel_add(float *proj1, float *proj, int na, int nb, float weight);
#endif // _KERNEL_ADD_CUH