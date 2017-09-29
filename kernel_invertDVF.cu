__global__ kernel(cudaArray *mx2, cudaTextureObject_t mx, int nx, int ny, int nz, const dim3 blockSize, int niter);

__host__ void kernel_invertDVF(cudaArray *d_mx2, cudaArray *d_my2, cudaArray *d_mz2, cudaTextureObject_t tex_mx, cudaTextureObject_t tex_my, cudaTextureObject_t tex_mz, int nx, int ny, int nz, int niter, const dim3 gridSize_img, const dim3 blockSize){
    kernel<<gridSize_img, blockSize>>>(d_mx2, tex_mx, nx, ny, nz, blockSize, niter);
    cudaDeviceSynchronize();
    kernel<<gridSize_img, blockSize>>>(d_my2, tex_my, nx, ny, nz, blockSize, niter);
    cudaDeviceSynchronize();
    kernel<<gridSize_img, blockSize>>>(d_mz2, tex_mz, nx, ny, nz, blockSize, niter);
    cudaDeviceSynchronize();
}

__global__ kernel(cudaArray *mx2, cudaTextureObject_t mx, int nx, int ny, int nz, const dim3, blockSize, niter);
{
    int x = blockSize.x * blockIdx.x + threadIdx.x;
    int y = blockSize.y * blockIdx.y + threadIdx.y;
    int z = blockSize.z * blockIdx.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz)
        return;
    float ix = 0;
    for (int iter = 0; iter < niter; iter ++){
        ix = tex3D<float>(mx, (ix + 0.5f), (iy + 0.5f), (iz + 0.5f)) * -1;
    }
    mx2[x][y][z] = ix;
}