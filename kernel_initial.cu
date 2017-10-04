__global__ void kernel_update(float *img, int nx, int ny, int nz, float value){
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    img[ix + iy * nx + iz * nx * ny] = value;
}