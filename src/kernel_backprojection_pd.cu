// past distance driven
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
    img[id] = -1.0f;return;
    float cphi, sphi, x1, y1, z1, xc, yc, zc, xc0, yc0, a, b;//  x20, y20, x2, y2, z2, x2n, y2n, z2n, x2m, y2m, p2x, p2y, p2z, p2xn, p2yn, p2zn, ptmp;
    // float ds, dt, temp, dst, det;
    // float xc, yc, zc, xcn, ycn, zcn, xcm, ycm, xc0, yc0;
    // float as, ae, bs, be, atmp, btmp, dsp, dtp, L;
    int ia, ib;
    angle += PI;
    cphi = (float)__cosf(angle);
    sphi = (float)__sinf(angle);

    x1 = -SO;
    y1 = 0.0f;
    z1 = 0.0f;
    xc = ix + 0.5f - nx / 2;
    yc = iy + 0.5f - ny / 2;
    zc = iz + 0.5f - nz / 2;
    xc0 = xc * cphi + yc * sphi;
    yc0 = -xc * sphi + yc * cphi;
    a = (xc0 - x1) / SD * yc0 / da - ai;
    b = (xc0 - x1) / SD * zc / db - bi;
    //if (a < 0 || a >= na)
    //    return;
    //if (b < 0 || b >= nb)
    //    return;
    ia = (int)floor(a);
    ib = (int)floor(b);
    img[id] -= ia + ib * na;//proj[ib + (na - 1 - ia) * nb];

}