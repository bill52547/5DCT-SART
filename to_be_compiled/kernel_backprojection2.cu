// pixel driven backprojection

#include <math.h>
#define ABS(x) ((x) > 0 ? (x) : - (x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define PI 3.141592653589793

__global__ void kernel_backprojection(float *img, float *proj, float angle, float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz){
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    img[id] = 0.0f;
    float xa, ya, za;
    xa = ix + 0.5f - nx / 2;
    ya = iy + 0.5f - ny / 2;
    za = iz + 0.5f - nz / 2;
    float cphi, sphi;
    angle += PI;
    cphi = (float)cosf(angle);
    sphi = (float)sinf(angle);
    float xa0, ya0, za0;
    xa0 = xa * cphi + ya * sphi;
    ya0 = -xa * sphi + ya * cphi;
    za0 = za;
    float x1, y1, z1;
    x1 = -SO;
    y1 = 0.0f;
    z1 = 0.0f;
    float a, b;
    a = ya0 / (xa0 - x1) * SD / da + na / 2;
    b = za0 / (xa0 - x1) * SD / db + nb / 2;
    int ia1, ia2, ib1, ib2;
    float wa1, wa2, wb1, wb2;
    if (a > 0.5f && a < na - 1.5f){
        if (b > 0.5f && b < nb - 1.5f){
            ia1 = (int)floor(a); ia2 = ia1 + 1; wa2 = a - ia1; wa1 = 1 - wa2;
            ib1 = (int)floor(b); ib2 = ib1 + 1; wb2 = b - ib1; wb1 = 1 - wb2;
            img[id] += proj[ia1 + ib1 * na] * wa1 * wb1 + 
                       proj[ia1 + ib2 * na] * wa1 * wb2 + 
                       proj[ia2 + ib1 * na] * wa2 * wb1 + 
                       proj[ia2 + ib2 * na] * wa2 * wb2;
        }
    }
    

}