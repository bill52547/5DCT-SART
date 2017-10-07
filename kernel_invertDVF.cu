__global__ void kernel_invertDVF(float *mx2, float *my2, float *mz2, cudaTextureObject_t alpha_x, cudaTextureObject_t alpha_y, cudaTextureObject_t alpha_z, cudaTextureObject_t beta_x, cudaTextureObject_t beta_y, cudaTextureObject_t beta_z, float volume, float flow, int nx, int ny, int nz, int niter)
{
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    float x = 0, y = 0, z = 0;
    for (int iter = 0; iter < niter; iter ++){
        x = - tex3D<float>(alpha_x, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f)) * volume
            - tex3D<float>(beta_x, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f)) * flow;
        y = - tex3D<float>(alpha_y, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f)) * volume
            - tex3D<float>(beta_y, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f)) * flow;
        z = - tex3D<float>(alpha_z, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f)) * volume
            - tex3D<float>(beta_z, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f)) * flow;
    }
    mx2[id] = x;
    my2[id] = y;
    mz2[id] = z;
}