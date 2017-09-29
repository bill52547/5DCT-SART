__global__ void kernel_update(cudaArray *img, int nx, int ny, int nz, float value){
    int x = blockSize.x * blockIdx.x + threadIdx.x;
    int y = blockSize.y * blockIdx.y + threadIdx.y;
    int z = blockSize.z * blockIdx.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz)
        return;
    img[x][y][z] = value;
}