#include <math.h>
#define ABS(x) ((x) > 0 ? (x) : - (x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

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
    cphi = (float)cosf(angle);
    sphi = (float)sinf(angle);
    float xl, yl, zl, xr, yr, zr, xt, yt, zt, xb, yb, zb; // x_Bleft...
    xl = (xa - 0.5f) * cphi + (ya - 0.5f) * sphi;
    yl = -(xa - 0.5f) * sphi + (ya - 0.5f) * cphi;
    zl = za;
    xr = (xa + 0.5f) * cphi + (ya + 0.5f) * sphi;
    yr = -(xa + 0.5f) * sphi + (ya + 0.5f) * cphi;
    zr = za;
    xt = xa * cphi + ya * sphi;
    yt = -xa * sphi + ya * cphi;
    zt = za + 0.5f;
    xb = xa * cphi + ya * sphi;
    yb = -xa * sphi + ya * cphi;
    zb = za - 0.5f;
    float bl, br, bt, bb, temp;// b_left on detector
    int ibl, ibr, ibt, ibb;
    bl = yl * SD / (xl + SO) / da;
    br = yr * SD / (xr + SO) / da;
    if (br < bl) {temp = br; br = bl; bl = temp;}
    bt = zt * SD / (xt + SO) / db;
    bb = zb * SD / (xb + SO) / db;
    if (bt < bb) {temp = bt; bt = bb; bb = temp;}
    ibl = (int)floor(bl + na / 2);
    ibr = (int)floor(br + na / 2);
    ibt = (int)floor(bt + nb / 2);
    ibb = (int)floor(bb + nb / 2);

    float inter_lr, inter_tb; // interval between boundary_left and boundary_right
    inter_lr = br - bl;
    inter_tb = bt - bb;
    float wa, wb;
    for (int ia = ibl; ia <= ibr; ia ++){
        if (ia < 0 || ia >= na)
            continue;
        wa = MIN(ia + 1, br + na / 2) - MAX(ia, bl + na / 2); wa /= inter_lr;
        for (int ib = ibb; ib <= ibt; ib ++){
            if (ib  < 0 || ib >= nb)
                continue;
            wb = MIN(ib + 1, bt + nb / 2) - MAX(ib, bb + nb / 2); wb /= inter_tb;
            img[id] += proj[ia + ib * na] * wa * wb;
        }
    }
}