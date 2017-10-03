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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define PROJ prhs[1]
#define GEO_PARA prhs[2]
#define ITER_PARA prhs[3]
#define OUT_IMG plhs[0]

// load parameters
// assume all the parameter are orginized as:
// dx = dy = dz = 1 
// da = db

// load geometry parameters, all need parameter for single view projection
int nx, ny, nz, na, nb, numImg, numBytesImg, numSingleProj, numBytesSingleProj;
float da, db, ai, bi, SO, SD;
nx = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nx"));
ny = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "ny"));
nz = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nz"));
numImg = nx * ny * nz; // size of image
numBytesImg = numImg * sizeof(float); // number of bytes in image
na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "na"));
nb = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nb"));
numSingleProj = na * nb;
numBytesSingleProj = numSingleProj * sizeof(float);
da = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "da"));
db = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "db"));
ai = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "ai"));
bi = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "bi"));
SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SO"));
SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SD"));

// load iterating parameters, for the whole bin
int n_view, n_iter, numProj, numBytesProj;
float *h_mx, *h_my, *h_mz, *h_mx2, *h_my2, *h_mz2, *angles, lambda;
n_view = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "nv")); // number of views in this bin
n_iter = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_iter")); // number of iterations of SART
h_mx = (float*)mxGetData(mxGetField(ITER_PARA, 0, "mx")); // index stead of difference
h_my = (float*)mxGetData(mxGetField(ITER_PARA, 0, "my")); 
h_mz = (float*)mxGetData(mxGetField(ITER_PARA, 0, "mz"));
h_mx2 = (float*)mxGetData(mxGetField(ITER_PARA, 0, "mx2")); // index stead of difference
h_my2 = (float*)mxGetData(mxGetField(ITER_PARA, 0, "my2")); 
h_mz2 = (float*)mxGetData(mxGetField(ITER_PARA, 0, "mz2"));
numProj = numSingleProj * n_view;
numBytesProj = numProj * sizeof(float);
angles = (float*)mxGetData(mxGetField(ITER_PARA, 0, "angles"));
lambda = (float)mxGetScalar(mxGetField(ITER_PARA, 0, "lambda"));

// load initial guess of image
float *h_img;
h_img = (float*)mxGetData(IN_IMG);

// load true projection value
float *h_proj;
h_proj = (float*)mxGetData(PROJ);

// define thread distributions
const dim3 gridSize_img((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (nz + BLOCKDEPTH - 1) / BLOCKDEPTH);
const dim3 gridSize_singleProj((na + BLOCKWIDTH - 1) / BLOCKWIDTH, (nb + BLOCKHEIGHT - 1) / BLOCKHEIGHT, 1);
const dim3 gridSize_proj((na + BLOCKWIDTH - 1) / BLOCKWIDTH, (nb + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (n_view + BLOCKDEPTH - 1) / BLOCKDEPTH);
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT, BLOCKDEPTH);

// CUDA 3DArray Malloc parameters
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
struct cudaExtent extent_proj = make_cudaExtent(na, nb, n_view);
struct cudaExtent extent_singleProj = make_cudaExtent(na, nb, 1);
const mwSize *outDim = mxGetDimensions(IN_IMG);

// malloc in device: DVF for SINGLE view image from the bin
float *d_mx, *d_my, *d_mz, *d_mx2, *d_my2, *d_mz2;
cudaMalloc((void**)&d_mx, numBytesImg);
cudaMalloc((void**)&d_my, numBytesImg);
cudaMalloc((void**)&d_mz, numBytesImg);
cudaMalloc((void**)&d_mx2, numBytesImg);
cudaMalloc((void**)&d_my2, numBytesImg);
cudaMalloc((void**)&d_mz2, numBytesImg);

// malloc in device: projection of the whole bin
float *d_proj;
cudaMalloc((void**)&d_proj, numBytesProj);

// copy to device: projection of the whole bin
cudaMemcpy(d_proj, h_proj, numBytesProj, cudaMemcpyHostToDevice);

// malloc in device: another projection pointer, with single view size
float *d_singleViewProj2;
cudaMalloc((void**)&d_singleViewProj2, numBytesSingleProj);

// malloc in device: projection of the whole bin
float *d_img;
cudaMalloc((void**)&d_img, numBytesImg);

// copy to device: initial guess of image
cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);

// malloc in device: another image pointer, for single view 
float *d_singleViewImg1, *d_singleViewImg2, *d_imgOnes;
cudaMalloc(&d_singleViewImg1, numBytesImg);
cudaMalloc(&d_singleViewImg2, numBytesImg);
cudaMalloc(&d_imgOnes, numBytesImg);


for (int iter = 0; iter < n_iter; iter++){ // iteration
    for (int i_view = 0; i_view < n_view; i_view++){ // view
        // memory copy to device of: DVF from bin reference image to i_view image
        // X
        cudaMemcpy(d_mx, h_mx + i_view * numBytesImg, numBytesImg, cudaMemcpyHostToDevice);

        // Y
        cudaMemcpy(d_my, h_my + i_view * numBytesImg, numBytesImg, cudaMemcpyHostToDevice);

        // Z
        cudaMemcpy(d_mz, h_mz + i_view * numBytesImg, numBytesImg, cudaMemcpyHostToDevice);

        // memory copy to device of: inverted DVF from bin reference image to i_view image
        // X
        cudaMemcpy(d_mx2, h_mx2 + i_view * numBytesImg, numBytesImg, cudaMemcpyHostToDevice);

        // Y
        cudaMemcpy(d_my2, h_my2 + i_view * numBytesImg, numBytesImg, cudaMemcpyHostToDevice);

        // Z
        cudaMemcpy(d_mz2, h_mz2 + i_view * numBytesImg, numBytesImg, cudaMemcpyHostToDevice);
        

        // deformed image for i_view, from reference image of the bin
        kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg1, d_img, d_mx2, d_my2, d_mz2, nx, ny, nz);
        cudaDeviceSynchronize();

        // projection of deformed image from initial guess
        kernel_projection<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_singleViewImg1, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz); // TBD
        cudaDeviceSynchronize();

        // difference between true projection and projection from initial guess
        // update d_singleViewProj2 instead of malloc a new one
        kernel_add<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_proj, i_view, na, nb, -1);
        cudaDeviceSynchronize();

        // backprojecting the difference of projections
        kernel_backprojection<<<gridSize_img, blockSize>>>(d_singleViewImg1, d_singleViewProj2, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
        cudaDeviceSynchronize();

        // deform backprojection back to the bin
        kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg2, d_singleViewImg1, d_mx, d_my, d_mz, nx, ny, nz);

        // calculate the ones backprojection data
        kernel_initial<<<gridSize_img, blockSize>>>(d_singleViewImg1, nx, ny, nz, 1);
        cudaDeviceSynchronize();
        kernel_projection<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_singleViewImg1, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
        cudaDeviceSynchronize();
        kernel_backprojection<<<gridSize_img, blockSize>>>(d_singleViewImg1, d_singleViewProj2, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
        cudaDeviceSynchronize();

        // weighting
        kernel_division<<<gridSize_img, blockSize>>>(d_singleViewImg2, d_singleViewImg1, nx, ny, nz);
        cudaDeviceSynchronize();
        
        // updating
        kernel_update<<<gridSize_img, blockSize>>>(d_img, d_singleViewImg2, nx, ny, nz, lambda);
        cudaDeviceSynchronize();              
    }
}
cudaFree(d_mx);
cudaFree(d_my);
cudaFree(d_mz);
cudaFree(d_mx2);
cudaFree(d_my2);
cudaFree(d_mz2);
// cudaFreeArray(d_img);
cudaFree(d_proj);
cudaFree(d_singleViewImg1);
cudaFree(d_singleViewImg2);
cudaFree(d_singleViewProj2);
OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
mxSetDimensions(OUT_IMG, outDim, 3);
mxSetData(OUT_IMG, mxMalloc(numBytesImg));
float *h_outimg = (float*)mxGetData(OUT_IMG);
cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);
cudaFree(d_img);
cudaDeviceReset();
return;
}

