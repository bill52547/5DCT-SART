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
#include "kernel_deformation.h"