
#ifndef _SART_CUDA_CUH
#define _SART_CUDA_CUH

#include "kernel_add.cuh" // kernel_add(d_proj1, d_proj, iv, na, nb, -1);
#include "kernel_division.cuh" // kernel_division(d_img1, d_img, nx, ny, nz);
#include "kernel_initial.cuh" // kernel_initial(img, nx, ny, nz, value);
#include "kernel_update.cuh" // kernel_update(d_img1, d_img, nx, ny, nz, lambda);
#include "kernel_projection.cuh" // kernel_projection(d_proj, d_img, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
#include "kernel_backprojection.cuh" // kernel_backprojection(d_img, d_proj, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
#include "kernel_deformation.cuh" // kernel_deformation(float *img1, float *img, float *mx2, float *my2, float *mz2, int nx, int ny, int nz);
#include "kernel_forwardDVF.cuh" // kernel_forwardDVF(float *mx, float *my, float *mz, cudaTextureObject_t alpha_x, cudaTextureObject_t alpha_y, cudaTextureObject_t alpha_z, 
#include "kernel_invertDVF.cuh"
#include "processBar.cuh"
#endif // _SART_CUDA_CUH