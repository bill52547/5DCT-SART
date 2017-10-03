// this program is try to do the SART program for a single bin
#include "universe_header.h"
#include "kernel_invertedDVF.h"
#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){

#include "kernel_add.h"
#include "kernel_backprojection.h"
#include "kernel_deformation.h"
#include "kernel_division.h"
#include "kernel_initial.h"
#include "kernel_invertDVF.h"
#include "kernel_projection.h"
#include "kernel_update.h"

// Macro for input and output
#define IN_IMG prhs[0]
#define PROJ prhs[1]
#define GEO_PARA prhs[2]
#define ITER_PARA prhs[3]
#define OUT_IMG plhs[0]
// #define OUT_VARIABLE plhs[1]

// load parameters
// assume all the parameter are orginized as:
// dx = dy = dz = 1 
// da = db

// load geometry parameters, all need parameter for single view projection
int nx, ny, nz, na, nb, numImg, numBytesImg, numSingleProj, numBytesSingleProj;
float da, db, ai, bi;
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
float *mx, *my, *mz, angle;
n_view = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "nv"));
n_iter = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_iter"));
mx = (float*)mxGetData(mxGetField(ITER_PARA, 0, "mx")); // index stead of difference
my = (float*)mxGetData(mxGetField(ITER_PARA, 0, "my"));
mz = (float*)mxGetData(mxGetField(ITER_PARA, 0, "mz"));
numProj = numSingleProj * n_view;
numBytesProj = numProj * sizeof(float);
angle = (float)mxGetScalar(mxGetField(ITER_PARA, 0, "angle"));

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

// CUDA 3DArray Copy parameters
cudaMemcpy3DParms copyParams = {0};
copyParams.extent = extent_proj;
copyParams.kind = cudaMemcpyHostToDevice;

// malloc in device: DVF for SINGLE view image from the bin
// for texture memory convenience, we malloc d_mx and d_mx2 as 3DArray
cudaArray *d_mx, *d_my, *d_mz, *d_mx2, *d_my2, *d_mz2;
cudaMalloc3DArray(&d_mx, &channelDesc, extent_img);
cudaMalloc3DArray(&d_my, &channelDesc, extent_img);
cudaMalloc3DArray(&d_mz, &channelDesc, extent_img);
cudaMalloc3DArray(&d_mx2, &channelDesc, extent_img);
cudaMalloc3DArray(&d_my2, &channelDesc, extent_img);
cudaMalloc3DArray(&d_mz2, &channelDesc, extent_img);

// malloc in device: projection of the whole bin
cudaArray *d_proj;
cudaMalloc3DArray(&d_proj, &channelDesc, extent_proj);

// copy to device: projection of the whole bin
cudaPitchedPtr projp;
projp = make_cudaPitchedPtr((void*) h_proj, na * sizeof(float), nb, nv);
copyParams.srcPtr = projp;
copyParams.dstArray = d_proj;
cudaMemcpy3D(&copyParams);

// malloc in device: another projection pointer, with single view size
cudaArray *d_singleViewProj2;
cudaMalloc3DArray(&d_singleViewProj2, numBytesSingleProj);

// malloc in device: projection of the whole bin
cudaArray *d_img;
cudaMalloc3DArray(&d_img, &channelDesc, extent_img);

// copy to device: initial guess of image
cudaPitchedPtr imgp;
imgp = make_cudaPitchedPtr((void*) h_img, nx * sizeof(float), ny, nz);
copyParams.extent = extent_img;
copyParams.srcPtr = imgp;
copyParams.dstArray = d_img;
cudaMemcpy3D(&copyParams);

// malloc in device: another image pointer, for single view 
cudaArray *d_singleViewImg1, *d_singleViewImg2, *d_imgOnes;
cudaMalloc3DArray(&d_singleViewImg1, &channelDesc, extent_img);
cudaMalloc3DArray(&d_singleViewImg2, &channelDesc, extent_img);
cudaMalloc3DArray(&d_imgOnes, &channelDesc, extent_img);

// cudaPitchedPtr for mx, my, mz
cudaPitchedPtr mxp, myp, mzp;
mxp = make_cudaPitchedPtr((void*) mx, nx * sizeof(float), ny, nz);
myp = make_cudaPitchedPtr((void*) my, nx * sizeof(float), ny, nz);
mzp = make_cudaPitchedPtr((void*) mz, nx * sizeof(float), ny, nz);


// Create texture object
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;


for (int iter = 0; iter < n_iter; i++){ // iteration
    for (int i_view = 0; i_view < n_view; i_view++){ // view
        // memory copy to device of: DVF from bin reference image to i_view image
        // X
        mxp = make_cudaPitchedPtr((void*)mx + i_view * numBytesImg, nx * sizeof(float), ny, nz);
        copyParams.srcPtr = mxp;
        copyParams.dstArray = d_mx;
        cudaMemcpy3D(&copyParams);

        // Y
        myp = make_cudaPitchedPtr((void*)my + i_view * numBytesImg, nx * sizeof(float), ny, nz);
        copyParams.srcPtr = myp;
        copyParams.dstArray = d_my;
        cudaMemcpy3D(&copyParams);

        // Z
        mzp = make_cudaPitchedPtr((void*)mz + i_view * numBytesImg, nx * sizeof(float), ny, nz);
        copyParams.srcPtr = mzp;
        copyParams.dstArray = d_mz;
        cudaMemcpy3D(&copyParams);

        // Create texture objects mx, my, mz
        // X
        resDesc.res.array.array = d_mx;
        cudaTextureObject_t tex_mx = 0;
        cudaCreateTextureObject(&tex_mx, &resDesc, &texDesc, NULL);
        // Y
        resDesc.res.array.array = d_my;
        cudaTextureObject_t tex_my = 0;
        cudaCreateTextureObject(&tex_my, &resDesc, &texDesc, NULL);
        // Z
        resDesc.res.array.array = d_mz;
        cudaTextureObject_t tex_mz = 0;
        cudaCreateTextureObject(&tex_mz, &resDesc, &texDesc, NULL);

        // find inverted DVF from forward DVF
        kernel_invertDVF(d_mx2, d_my2, d_mz2, tex_mx, tex_my, tex_mz, nx, ny, nz, 10, gridSize_img, blockSize);
        cudaDeviceSynchronize();

        // Create texture object tex_img
        resDesc.res.array.array = d_img;
        cudaTextureObject_t tex_img = 0;
        cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);

        // deformed image for i_view, from reference image of the bin
        kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg1, tex_img, d_mx2, d_my2, d_mz2, nx, ny, nz);
        cudaDeviceSynchronize();

        // projection of deformed image from initial guess
        kernel_projection<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_singleViewImg1); // TBD
        cudaDeviceSynchronize();

        // difference between true projection and projection from initial guess
        // update d_singleViewProj2 instead of malloc a new one
        kernel_add<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_proj, i_view, na, nb, -1);
        cudaDeviceSynchronize();

        // backprojecting the difference of projections
        kernel_backprojection<<<gridSize_img, blockSize>>>(d_singleViewImg1, d_singleViewProj2, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
        cudaDeviceSynchronize();

        // Create texture object tex_img1
        resDesc.res.array.array = d_singleViewImg1;
        cudaTextureObject_t tex_img1 = 0;
        cudaCreateTextureObject(&tex_img1, &resDesc, &texDesc, NULL);

        // deform backprojection back to the bin
        kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg2, tex_img1, d_mx, d_my, d_mz, nx, ny, nz);

        // calculate the ones backprojection data
        kernel_kernel<<<gridSize_img, blockSize>>>(d_singleViewImg1, nx, ny, nz, 1);
        cudaDeviceSynchronize();
        kernel_projection<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_singleViewImg1, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
        cudaDeviceSynchronize();
        kernel_backprojection<<<gridSize_img, blockSize>>>(d_singleViewImg1, d_singleViewProj2, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
        cudaDeviceSynchronize();

        // weighting
        kernel_division<<<gridSize_img, blockSize>>>(d_singleViewImg2, d_singleViewImg1, nx, ny, nz);
        cudaDeviceSynchronize();
        
        // updating
        kernel_update<<<gridSize_img, blockSize>>>(d_img, d_singleViewImg1, nx, ny, nz, lambda);
        cudaDeviceSynchronize();

        cudaDestroyTextureObject(tex_mx);
        cudaDestroyTextureObject(tex_my);
        cudaDestroyTextureObject(tex_mz);
        cudaDestroyTextureObject(tex_img);
        cudaDestroyTextureObject(tex_img1);                  
    }
}
cudaFreeArray(d_mx);
cudaFreeArray(d_my);
cudaFreeArray(d_mz);
cudaFreeArray(d_mx2);
cudaFreeArray(d_my2);
cudaFreeArray(d_mz2);
// cudaFreeArray(d_img);
cudaFreeArray(d_imgOnes);
cudaFreeArray(d_proj);
cudaFreeArray(d_singleViewImg1);
cudaFreeArray(d_singleViewImg2);
cudaFreeArray(d_singleViewProj2);
OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS,mxREAL);
mxSetDimensions(OUT_IMG, extent_img, 3);
mxSetData(OUT_IMG, mxMalloc(numBytesImg));
float *h_outimg = (float*)mxGetData(OUT_IMG);
cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);
cudaFreeArray(d_img);
}