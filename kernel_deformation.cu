__global__ void kernel_deformation(float *img1, float *img, float *mx2, float *my2, float *mz2, int nx, int ny, int nz){
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    int xi = mx2[id];
    int yi = my2[id];
    int zi = mz2[id];
    img1[id] = 0.0f;
    if (xi < 0.5f || xi >= nx - 0.5f || yi < 0.5f || yi >= ny - 0.5f || zi < 0.5f || zi >= nz - 0.5f)
        return;
    int ix1, ix2, iy1, iy2, iz1, iz2;
    float wx1, wx2, wy1, wy2, wz1, wz2;
    ix1 = (int)floor(xi - 0.5f); ix2 = ix1 + 1; wx2 = xi - 0.5f - ix1; wx1 = 1.0f - wx2;
    iy1 = (int)floor(yi - 0.5f); iy2 = iy1 + 1; wy2 = yi - 0.5f - iy1; wy1 = 1.0f - wy2;
    iz1 = (int)floor(zi - 0.5f); iz2 = iz1 + 1; wz2 = zi - 0.5f - iz1; wz1 = 1.0f - wz2;
    img1[id] += img[ix1 + iy1 * nx + iz1 * nx * ny] * wx1 * wy1 * wz1;
    img1[id] += img[ix1 + iy1 * nx + iz2 * nx * ny] * wx1 * wy1 * wz2;
    img1[id] += img[ix1 + iy2 * nx + iz1 * nx * ny] * wx1 * wy2 * wz1;
    img1[id] += img[ix1 + iy2 * nx + iz2 * nx * ny] * wx1 * wy2 * wz2;
    img1[id] += img[ix2 + iy1 * nx + iz1 * nx * ny] * wx2 * wy1 * wz1;
    img1[id] += img[ix2 + iy1 * nx + iz2 * nx * ny] * wx2 * wy1 * wz2;
    img1[id] += img[ix2 + iy2 * nx + iz1 * nx * ny] * wx2 * wy2 * wz1;
    img1[id] += img[ix2 + iy2 * nx + iz2 * nx * ny] * wx2 * wy2 * wz2;
}


//     int x = blockSize.x * blockIdx.x + threadIdx.x;
//     int y = blockSize.y * blockIdx.y + threadIdx.y;
//     int z = blockSize.z * blockIdx.z + threadIdx.z;
//     if (x >= nx || y >= ny || z >= nz)
//         return;
//     int xi = mx2[x][y][z];
//     int yi = my2[x][y][z];
//     int zi = mz2[x][y][z];

//     singleViewImg1[x][y][z] = tex3D<float>(tex_img, xi-0.5f, yi-0.5f, zi-0.5f);
// }
