__global__ void kernel_division(float *img1, float *img, int nx, int ny, int nz)
{
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    if (img[id] < 0.0001)
        img1[id] = 0;
    else
        img1[id] /= img[id];
}
