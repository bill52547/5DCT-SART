__global__ void kernel_deformation(cudaArray *singleViewImg1, cudaTextureObject_t tex_img, cudaArray *mx2, cudaArray *my2, cudaArray *mz2, int nx, int ny, int nz){
    int x = blockSize.x * blockIdx.x + threadIdx.x;
    int y = blockSize.y * blockIdx.y + threadIdx.y;
    int z = blockSize.z * blockIdx.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz)
        return;
    int xi = mx2[x][y][z];
    int yi = my2[x][y][z];
    int zi = mz2[x][y][z];

    singleViewImg1[x][y][z] = tex3D<float>(tex_img, xi-0.5f, yi-0.5f, zi-0.5f);
}
