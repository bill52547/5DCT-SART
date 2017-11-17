#include "kernel_update.cuh"
__global__ void kernel_update(float *img1, float *img, int nx, int ny, int nz, float lambda){
    int ix = BLOCKWIDTH * blockIdx.x + threadIdx.x;
    int iy = BLOCKHEIGHT * blockIdx.y + threadIdx.y;
    int iz = BLOCKDEPTH * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    img1[id] -= lambda * img[id];
    if (img1[id] < 0.0f)
        img1[id] = 0.0f;
    if (img1[id] > 2500.0f)
        img1[id] = 0.0f;
}
