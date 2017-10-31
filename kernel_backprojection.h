#ifndef _KERNEL_BACKPROJECTION_H
#define _KERNEL_BACKPROJECTION_H
#include <math.h>
// #include "host_create_texture_object.h"
#ifndef _UNIVERSAL
#define _UNIVERSAL
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(x) ((x) > 0 ? (x) : -(x))
#define PI 3.141592653589793f

#define BLOCKWIDTH 16
#define BLOCKHEIGHT 16 
#define BLOCKDEPTH 4
#endif
__host__ void kernel_backprojection(float *img, float *proj, float angle, float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz);
__global__ void kernel(float *img, cudaTextureObject_t tex_proj, float angle, float SO, float SD, int nu, int nv, float du, float dv, float ui, float vi, int nx, int ny, int nz);
__device__ void get_uv_ranges(float* uMin, float* uMax, float* vMin, float* vMax, float* uv, float* dd_intersection, float* a1, float* a2, float* z1, float* z2, float* dd_sourcePosition, float* dd_centralDetectorPosition, float* u1, float* v1, float* dd_helical_detector_vector);
__device__ void get_intersection_line_plane(float* intersection, float* p1, float* p2, float* x0, float* n);
__device__ void get_uv_ind_flat(int* uvInd, float u, float v, float du, float dv, int nu, int nv, float ui, float vi);
#endif