#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "kernel_division.h"

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
float *img, *img1;
int nx, ny, nz;
img1 = (float*)mxGetData(prhs[0]);
img = (float*)mxGetData(prhs[1]);
nx = (int)mxGetScalar(prhs[2]);
ny = (int)mxGetScalar(prhs[3]);
nz = (int)mxGetScalar(prhs[4]);

float *d_img1, *d_img;
cudaMalloc((void**)&d_img, nx * ny * nz * sizeof(float));
cudaMalloc((void**)&d_img1, nx * ny * nz * sizeof(float));
cudaMemcpy(d_img, img, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_img1, img1, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);
const dim3 gridSize((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (nz + BLOCKDEPTH - 1) / BLOCKDEPTH);
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT, BLOCKDEPTH);
kernel_division<<<gridSize, blockSize>>>(d_img1, d_img, nx, ny, nz);
cudaMemcpy(img1, d_img1, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(d_img);
cudaFree(d_img1);
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

// __global__ void kernel_add(cudaArray *img1, cudaArray *img, int iv, int nx, int ny, float weight);

// void mexFunction(int nlhs, mxArray *plhs[],
//                  int nrhs, mxArray const *prhs[]){
// float *img, *img1;
// int nx, ny, iv;
// img1 = (float*)mxGetData(prhs[0]);
// img = (float*)mxGetData(prhs[1]);
// iv = (int)mxGetScalar(prhs[2]);
// nx = (int)mxGetScalar(prhs[3]);
// ny = (int)mxGetScalar(prhs[4]);

// // pitched memory
// cudaPitchedPtr p_img, p_img1;
// p_img = make_cudaPitchedPtr((void*)img, nx * sizeof(float), ny, 1);
// p_img1 = make_cudaPitchedPtr((void*)img1, nx * sizeof(float), ny, 1);

// cudaArray *d_img1, *d_img;
// cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
// struct cudaExtent extent = make_cudaExtent(nx, ny, 1);
// cudaMalloc3DArray(&d_img, &channelDesc, extent);
// cudaMalloc3DArray(&d_img1, &channelDesc, extent);
// cudaMemcpy3DParms copyParams = {0};
// copyParams.extent = extent;
// copyParams.kind = cudaMemcpyHostToDevice;
// copyParams.srcPtr = p_img;
// copyParams.dstArray = d_img;
// cudaMemcpy3D(&copyParams);
// copyParams.srcPtr = p_img1;
// copyParams.dstArray = d_img1;
// cudaMemcpy3D(&copyParams);

// const dim3 gridSize((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (1 + BLOCKDEPTH - 1) / BLOCKDEPTH);
// const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT, BLOCKDEPTH);
// kernel_add<<<gridSize, blockSize>>>(d_img1, d_img, iv, nx, ny, -1);
// cudaMemcpy(img1, d_img1, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
// cudaFreeArray(d_img);
// cudaFreeArray(d_img1);
// //Reset device for profiling
// cudaDeviceReset();
// return;
// }