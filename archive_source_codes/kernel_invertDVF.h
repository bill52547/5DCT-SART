__host__ void kernel_invertDVF(cudaArray *d_mx2, cudaArray *d_my2, cudaArray *d_mz2, cudaTextureObject_t tex_mx, cudaTextureObject_t tex_my, cudaTextureObject_t tex_mz, int nx, int ny, int nz, int niter, const dim3 gridSize_img, const dim3 blockSize);