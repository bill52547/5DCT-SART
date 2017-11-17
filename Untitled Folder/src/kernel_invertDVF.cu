#include "kernel_invertDVF.cuh"
__global__ void kernel_invertDVF(float *mx2, float *my2, float *mz2, cudaTextureObject_t mx, cudaTextureObject_t my, cudaTextureObject_t mz, int nx, int ny, int nz, int niter)
{
    int ix = BLOCKWIDTH * blockIdx.x + threadIdx.x;
    int iy = BLOCKHEIGHT * blockIdx.y + threadIdx.y;
    int iz = BLOCKDEPTH * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    float x = 0.0f, y = 0.0f, z = 0.0f;
    for (int iter = 0; iter < niter; iter ++){
        x = - tex3D<float>(mx, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f));
        y = - tex3D<float>(my, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f));
        z = - tex3D<float>(mz, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f));
    }
    mx2[id] = x;
    my2[id] = y;
    mz2[id] = z;
}