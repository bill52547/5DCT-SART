__global__ kernel(float *mx2, float *mx, int nx, int ny, int nz, int niter);

__host__ void kernel_invertDVF(float *d_mx2, float *d_my2, float *d_mz2, float *tex_mx, float *tex_my, float *tex_mz, int nx, int ny, int nz, int niter, const dim3 gridSize_img, const dim3 blockSize){
    kernel<<gridSize_img, blockSize>>>(d_mx2, tex_mx, nx, ny, nz, blockSize, niter);
    cudaDeviceSynchronize();
}

__global__ kernel(float *mx2, float *my2, float *mz2, float *mx, float *my, float *mz, int nx, int ny, int nz, int niter);
{
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    float x = 0, y = 0, z = 0;
    int ix1, ix2, iy1, iy2, iz1, iz2;
    float wx1, wx2, wy1, wy2, wz1, wz2;
    for (int iter = 0; iter < niter; iter ++){
        ix1 = (int)floor(xi + 0.5f + ix); ix2 = ix1 + 1; wx2 = xi + 0.5f + ix - ix1; wx1 = 1.0f - wx2;
        iy1 = (int)floor(yi + 0.5f + iy); iy2 = iy1 + 1; wy2 = yi + 0.5f + iy - iy1; wy1 = 1.0f - wy2;
        iz1 = (int)floor(zi + 0.5f + iz); iz2 = iz1 + 1; wz2 = zi + 0.5f + iz - iz1; wz1 = 1.0f - wz2;
        img1[id] += img[ix1 + iy1 * nx + iz1 * nx * ny] * wx1 * wy1 * wz1;
        img1[id] += img[ix1 + iy1 * nx + iz2 * nx * ny] * wx1 * wy1 * wz2;
        img1[id] += img[ix1 + iy2 * nx + iz1 * nx * ny] * wx1 * wy2 * wz1;
        img1[id] += img[ix1 + iy2 * nx + iz2 * nx * ny] * wx1 * wy2 * wz2;
        img1[id] += img[ix2 + iy1 * nx + iz1 * nx * ny] * wx2 * wy1 * wz1;
        img1[id] += img[ix2 + iy1 * nx + iz2 * nx * ny] * wx2 * wy1 * wz2;
        img1[id] += img[ix2 + iy2 * nx + iz1 * nx * ny] * wx2 * wy2 * wz1;
        img1[id] += img[ix2 + iy2 * nx + iz2 * nx * ny] * wx2 * wy2 * wz2;
        x = tex3D<float>(mx, (x + 0.5f), (y + 0.5f), (z + 0.5f)) * -1;
    }
    mx2[id] = x;
}