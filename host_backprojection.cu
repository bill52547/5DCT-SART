#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "kernel_backprojection.h"

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(x) ((x) > 0 ? (x) : -(x))
#define PI 3.141592653589793
// Set thread block size
#define BLOCKWIDTH 16
#define BLOCKHEIGHT 16 
#define BLOCKDEPTH 4

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]){
float *h_proj, *h_img;
h_img = (float*)mxGetData(prhs[0]);
h_proj = (float*)mxGetData(prhs[1]);

int na, nb, nx, ny, nz;
float angle, SO, SD, da, ai, db, bi, dx;
angle = (float)mxGetScalar(mxGetField(prhs[2], 0, "angle"));
SO = (float)mxGetScalar(mxGetField(prhs[2], 0, "SO"));
SD = (float)mxGetScalar(mxGetField(prhs[2], 0, "SD"));
da = (float)mxGetScalar(mxGetField(prhs[2], 0, "da"));
db = (float)mxGetScalar(mxGetField(prhs[2], 0, "db"));
ai = (float)mxGetScalar(mxGetField(prhs[2], 0, "ai"));
bi = (float)mxGetScalar(mxGetField(prhs[2], 0, "bi"));
na = (int)mxGetScalar(mxGetField(prhs[2], 0, "na"));
nb = (int)mxGetScalar(mxGetField(prhs[2], 0, "nb"));
nx = (int)mxGetScalar(mxGetField(prhs[2], 0, "nx"));
ny = (int)mxGetScalar(mxGetField(prhs[2], 0, "ny"));
nz = (int)mxGetScalar(mxGetField(prhs[2], 0, "nz"));

float *d_proj, *d_img;
cudaMalloc((void**)&d_img, nx * ny * nz * sizeof(float));
cudaMalloc((void**)&d_proj, na * nb * sizeof(float));
cudaMemcpy(d_img, h_img, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_proj, h_proj, na * nb * sizeof(float), cudaMemcpyHostToDevice);

const dim3 gridSize((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (nz + BLOCKDEPTH - 1) / BLOCKDEPTH);
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT, BLOCKDEPTH);
kernel_backprojection(d_img, d_proj, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
cudaMemcpy(h_img, d_img, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(d_proj);
cudaFree(d_img);
//Reset device for profiling
cudaDeviceReset();
return;
}


// #include "mex.h"
// #include "matrix.h"
// #include "gpu/mxGPUArray.h"
// #include <stdlib.h>
// #include <cuda_runtime.h>
// #include <math.h>
// #include <iostream>
// // #include "kernel_add.h"

// #define MAX(a,b) (((a) > (b)) ? (a) : (b))
// #define MIN(a,b) (((a) < (b)) ? (a) : (b))
// #define ABS(x) ((x) > 0 ? (x) : -(x))
// #define PI 3.141592653589793
// // Set thread block size
// #define BLOCKWIDTH 16
// #define BLOCKHEIGHT 16 
// #define BLOCKDEPTH 4

// __global__ void kernel_add(cudaArray *proj1, cudaArray *proj, int iv, int na, int nb, float weight);

// void mexFunction(int nlhs, mxArray *plhs[],
//                  int nrhs, mxArray const *prhs[]){
// float *h_proj, *h_proj1;
// int na, nb, iv;
// h_proj1 = (float*)mxGetData(prhs[0]);
// h_proj = (float*)mxGetData(prhs[1]);
// iv = (int)mxGetScalar(prhs[2]);
// na = (int)mxGetScalar(prhs[3]);
// nb = (int)mxGetScalar(prhs[4]);

// // pitched memory
// cudaPitchedPtr p_proj, p_proj1;
// p_proj = make_cudaPitchedPtr((void*)h_proj, na * sizeof(float), nb, 1);
// p_proj1 = make_cudaPitchedPtr((void*)h_proj1, na * sizeof(float), nb, 1);

// cudaArray *d_proj1, *d_proj;
// cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
// struct cudaExtent extent = make_cudaExtent(na, nb, 1);
// cudaMalloc3DArray(&d_proj, &channelDesc, extent);
// cudaMalloc3DArray(&d_proj1, &channelDesc, extent);
// cudaMemcpy3DParms copyParams = {0};
// copyParams.extent = extent;
// copyParams.kind = cudaMemcpyHostToDevice;
// copyParams.srcPtr = p_proj;
// copyParams.dstArray = d_proj;
// cudaMemcpy3D(&copyParams);
// copyParams.srcPtr = p_proj1;
// copyParams.dstArray = d_proj1;
// cudaMemcpy3D(&copyParams);

// const dim3 gridSize((na + BLOCKWIDTH - 1) / BLOCKWIDTH, (nb + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (1 + BLOCKDEPTH - 1) / BLOCKDEPTH);
// const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT, BLOCKDEPTH);
// kernel_add<<<gridSize, blockSize>>>(d_proj1, d_proj, iv, na, nb, -1);
// cudaMemcpy(h_proj1, d_proj1, na * nb * sizeof(float), cudaMemcpyDeviceToHost);
// cudaFreeArray(d_proj);
// cudaFreeArray(d_proj1);
// //Reset device for profiling
// cudaDeviceReset();
// return;
// }