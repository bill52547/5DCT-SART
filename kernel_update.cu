__global__ void kernel_update(cudaArray *img1, cudaArray *img, int nx, int ny, int nz, float lambda){
    int x = blockSize.x * blockIdx.x + threadIdx.x;
    int y = blockSize.y * blockIdx.y + threadIdx.y;
    int z = blockSize.z * blockIdx.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz)
        return;
    img1[x][y][z] -= lambda * img[x][y][z];
    if (img1[x][y][z] < 0)
        img1[x][y][z] = 0;
}
