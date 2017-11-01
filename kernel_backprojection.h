#ifndef _KERNEL_BACKPROJECTION_H
#define _KERNEL_BACKPROJECTION_H
#include <math.h>
#include "mex.h"
// #include "host_create_texture_object.h"
// #ifndef _UNIVERSAL
// #define _UNIVERSAL
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
//#define MAX4(a, b, c, d) (((((a) > (b)) ? (a) : (b)) > (((c) > (d)) ? (c) : (d))) > (((a) > (b)) ? (a) : (b)) : (((c) > (d)) ? (c) : (d)))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))
#define ABS(x) ((x) > 0 ? (x) : -(x))
#define PI 3.141592653589793f

#define BLOCKWIDTH 16
#define BLOCKHEIGHT 16 
#define BLOCKDEPTH 4
// #endif
__host__ void kernel_backprojection(float *d_img, float *d_proj, float angle,float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz);
__global__ void kernel(float *img, cudaTextureObject_t tex_proj, float angle, float SO, float SD, int nu, int nv, float du, float dv, float ui, float vi, int nx, int ny, int nz);
#endif