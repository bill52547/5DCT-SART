#include "kernel_initial.cuh"
__global__ void kernel_initial(float *img, int nx, int ny, int nz, float value){
    int ix = BLOCKWIDTH * blockIdx.x + threadIdx.x;
    int iy = BLOCKHEIGHT * blockIdx.y + threadIdx.y;
    int iz = BLOCKDEPTH * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    img[ix + iy * nx + iz * nx * ny] = value;
}