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
    img[id] = 0.0f;
    float cphi, sphi;
    angle += PI;
    cphi = (float)cosf(angle);
    sphi = (float)sinf(angle);
    float xa, ya, za;
    xa = ix - nx / 2 + 0.5f;
    ya = iy - ny / 2 + 0.5f;
    za = iz - nz / 2 + 0.5f;
    // xa = xa0 * cphi - ya0 * sphi;
    // ya = xa0 * sphi + ya0 * cphi;
    float xl, yl, zl, xr, yr, zr, xt, yt, zt, xb, yb, zb;
    xl = (xa - 0.5f) * cphi + (ya - 0.5f) * sphi;
    yl = -(xa - 0.5f) * sphi + (ya - 0.5f) * cphi;
    xr = (xa + 0.5f) * cphi + (ya + 0.5f) * sphi;
    yr = -(xa + 0.5f) * sphi + (ya + 0.5f) * cphi;
    zl = za; zr = za;
    xt = xa * cphi + ya * sphi;
    yt = -xa * sphi + ya * cphi;
    zt = za + 0.5f;
    xb = xa * cphi + ya * sphi;
    yb = -xa * sphi + ya * cphi;
    zb = za - 0.5f;
    float bl, br, bt, bb, temp, wlr, wbt;
    bl = yl * SD / (SO + xl) / da + na / 2;
    br = yr * SD / (SO + xr) / da + na / 2;
    if (bl > br) {temp = bl; bl = br; br = temp;}
    bt = zt * SD / (SO * xt) / db + nb / 2;
    bb = zb * SD / (SO * xb) / db + nb / 2;
    wlr = br - bl;
    wbt = bt - bb;

    int ibl, ibr, ibt, ibb;
    ibl = (int)floor(bl);
    ibr = (int)floor(br);
    ibt = (int)floor(bt);
    ibb = (int)floor(bb);
    float w1, w2;
    for (int ia = ibl; ia <= ibr; ia ++){
        if (ia < 0 || ia >= na)
            continue;
        for (int ib = ibb; ib <= ibt; ib++){
            if (ib < 0 || ib >= nb)
                continue;
            w1 = MIN(br, ia + 1) - MAX(bl, ia);
            w2 = MIN(bt, ib + 1) - MAX(bb, ib);
            img[id] += proj[ia + ib * na] * w1 * w2 / wlr / wbt;
        }   
    }
}