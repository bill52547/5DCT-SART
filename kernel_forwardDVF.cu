__global__ void kernel_forwardDVF(float *mx, float *my, float *mz, cudaTextureObject_t alpha_x, cudaTextureObject_t alpha_y, cudaTextureObject_t alpha_z, cudaTextureObject_t beta_x, cudaTextureObject_t beta_y, cudaTextureObject_t beta_z, cudaTextureObject_t const_x, cudaTextureObject_t const_y, cudaTextureObject_t const_z, float volume, float flow, int nx, int ny, int nz)
{
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;    
    mx[id] = tex3D<float>(alpha_x, (ix + 0.5f), (iy + 0.5f), (iz + 0.5f)) * volume
           + tex3D<float>(beta_x, (ix + 0.5f), (iy + 0.5f), (iz + 0.5f)) * flow
           + tex3D<float>(const_x, (ix + 0.5f), (iy + 0.5f), (iz + 0.5f));
    my[id] = tex3D<float>(alpha_y, (ix + 0.5f), (iy + 0.5f), (iz + 0.5f)) * volume
           + tex3D<float>(beta_y, (ix + 0.5f), (iy + 0.5f), (iz + 0.5f)) * flow
           + tex3D<float>(const_y, (ix + 0.5f), (iy + 0.5f), (iz + 0.5f));
    mz[id] = tex3D<float>(alpha_z, (ix + 0.5f), (iy + 0.5f), (iz + 0.5f)) * volume
           + tex3D<float>(beta_z, (ix + 0.5f), (iy + 0.5f), (iz + 0.5f)) * flow
           + tex3D<float>(const_z, (ix + 0.5f), (iy + 0.5f), (iz + 0.5f));

}