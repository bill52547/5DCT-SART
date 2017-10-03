__global__ void kernel_division(cudaArray *img1, cudaArray *img, int nx, int ny, int nz)
{
    int x = blockSize.x * blockIdx.x + threadIdx.x;
    int y = blockSize.y * blockIdx.y + threadIdx.y;
    int z = blockSize.z * blockIdx.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz)
        return;
    if (img[x][y][z] == 0)
        img1[x][y][z] = 0;
    else
        img1[x][y][z] /= img[x][y][z];
}
