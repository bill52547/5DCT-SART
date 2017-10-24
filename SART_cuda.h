// this program is try to do the SART program for a single bin
// #include "universe_header.h"
#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(x) ((x) > 0 ? (x) : -(x))
#define PI 3.141592653589793
// Set thread block size
#define BLOCKWIDTH 16
#define BLOCKHEIGHT 16 
#define BLOCKDEPTH 4

#include "kernel_add.h" // kernel_add(d_proj1, d_proj, iv, na, nb, -1);
#include "kernel_division.h" // kernel_division(d_img1, d_img, nx, ny, nz);
#include "kernel_initial.h" // kernel_initial(img, nx, ny, nz, value);
#include "kernel_update.h" // kernel_update(d_img1, d_img, nx, ny, nz, lambda);
#include "kernel_projection.h" // kernel_projection(d_proj, d_img, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
#include "kernel_backprojection.h" // kernel_backprojection(d_img, d_proj, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
#include "kernel_deformation.h" // kernel_deformation(float *img1, float *img, float *mx2, float *my2, float *mz2, int nx, int ny, int nz);
#include "kernel_forwardDVF.h" // kernel_forwardDVF(float *mx, float *my, float *mz, cudaTextureObject_t alpha_x, cudaTextureObject_t alpha_y, cudaTextureObject_t alpha_z, cudaTextureObject_t beta_x, cudaTextureObject_t beta_y, cudaTextureObject_t beta_z,   cudaTextureObject_t const_x, cudaTextureObject_t const_y, cudaTextureObject_t const_z, float volume, float flow, int nx, int ny, int nz);
#include "kernel_invertDVF.h" //__global__ kernel_invertedDVF(float *mx2, float *my2, float *mz2, cudaTextureObject_t alpha_x, cudaTextureObject_t alpha_y, cudaTextureObject_t alpha_z, cudaTextureObject_t beta_x, cudaTextureObject_t beta_y, cudaTextureObject_t beta_z, int nx, int ny, int nz, int niter);
// #include "dist_cuda_functions.h"
// #include 

