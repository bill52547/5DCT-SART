#include "kernel_division.cuh"
__global__ void kernel_division(float *img1, float *img, int nx, int ny, int nz)
{
    int ix = BLOCKWIDTH * blockIdx.x + threadIdx.x;
    int iy = BLOCKHEIGHT * blockIdx.y + threadIdx.y;
    int iz = BLOCKDEPTH * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;

    if (img[id] > 0.01f)
        img1[id] /= img[id];

    // if (isnan(img1[id]))
    //     img1[id] = 0.0f;
    // if (isinf(img1[id]))
    //     img1[id] = 0.0f;    
}
