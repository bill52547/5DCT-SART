#ifndef _KERNEL_PROJECTION_H
#define _KERNEL_PROJECTION_H
#include "universal.cuh"
__global__ void kernel_projection(float *proj, float *img, float angle, float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz);

#endif // _KERNEL_PROJECTION_H
