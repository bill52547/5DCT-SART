#ifndef _KERNEL_BACKPROJECTION_CUH
#define _KERNEL_BACKPROJECTION_CUH
#include "universal.cuh"
__host__ void kernel_backprojection(float *d_img, float *d_proj, float angle,float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz);
__global__ void kernel(float *img, cudaTextureObject_t tex_proj, float angle, float SO, float SD, int nu, int nv, float du, float dv, float ui, float vi, int nx, int ny, int nz);
#endif // _KERNEL_BACKPROJECTION_CUH