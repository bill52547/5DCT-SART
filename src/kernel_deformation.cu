#include "kernel_deformation.cuh"
__global__ void kernel_deformation(float *img1, cudaTextureObject_t tex_img, float *mx2, float *my2, float *mz2, int nx, int ny, int nz){
    int ix = BLOCKWIDTH * blockIdx.x + threadIdx.x;
    int iy = BLOCKHEIGHT * blockIdx.y + threadIdx.y;
    int iz = BLOCKDEPTH * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = iy + ix * ny + iz * nx * ny;
    float xi = iy + my2[id];
    float yi = ix + mx2[id];
    float zi = iz + mz2[id];
    img1[id] = tex3D<float>(tex_img, xi + 0.5f, yi + 0.5f, zi + 0.5f);
}