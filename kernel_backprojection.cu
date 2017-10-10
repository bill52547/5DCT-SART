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
    float cphi, sphi,x1, y1, z1, x20, y20, x2, y2, z2, x2n, y2n, z2n, x2m, y2m, p2x, p2y, p2z, p2xn, p2yn, p2zn, ptmp;
    float ds, dt, temp, dst, det;
    float xc, yc, zc, xcn, ycn, zcn, xcm, ycm, xc0, yc0;
    float as, ae, bs, be, atmp, btmp, dsp, dtp, L;
    angle += PI;
    cphi = (float)cosf(angle);
    sphi = (float)sinf(angle);

    x1 = -SO * cphi;
    y1 = -SO * sphi;
    z1 = 0.0f;
    if (ABS(sphi) < 0.7071)
    {
        xc = ix - nx / 2 + 0.5f;

        yc = iy - ny / 2;
        ycn = iy - ny / 2 + 1.0f;
        ycm = (yc + ycn) / 2;
        xc0 = cphi * xc + sphi * yc + SO;
        yc0 = -sphi * xc + cphi * yc;
        as = yc0 / xc0 * SD / da - ai + 0.5f;
        xc0 = cphi * xc + sphi * ycn + SO;
        yc0 = -sphi * xc + cphi * ycn;
        ae = yc0 / xc0 * SD / da - ai + 0.5f;
        if (as > ae)
        {atmp = as; as = ae; ae = atmp;}
        if (ae < 0.0f || as >= (float)na)
            return;
        if (as < 0.0f)
            as = 0.0f;
        if (ae >= na)
            ae = float(na);

        zc = iz - nz / 2;
        zcn = iz - nz / 2 + 1.0f;
        xc0 = cphi * xc + sphi * ycm;
        yc0 = -sphi * xc + cphi * ycm;
        L = (float)sqrt((xc0 + SO) * (xc0 + SO) + yc0 * yc0);
        bs = (zc - z1) / L * SD / db - bi + 0.5f;
        be = (zcn - z1) / L * SD / db - bi + 0.5f;

        if (bs > be)
        {btmp = bs; bs = be; be = btmp;}
        if (be < 0.0f || bs >= (float)nb)
            return;
        if (bs < 0.0f)
            bs = 0.0f;
        if (be >= nb)
            be = float(nb);
        for (int ia = (int)floor(as); ia < (int)ceil(ae); ia++)
        {
            x20 = SD - SO;
            y20 = (ia + ai - 0.5f) * da;
            x2 = x20 * cphi - y20 * sphi;
            y2 = x20 * sphi + y20 * cphi;
            x20 = SD - SO;
            y20 = (ia + ai + 0.5f) * da;
            x2n = x20 * cphi - y20 * sphi;
            y2n = x20 * sphi + y20 * cphi;
            x2m = (x2 + x2n) / 2;
            y2m = (y2 + y2n) / 2;
            temp = (y2 - y1) / (x2 - x1);
            p2y = (ix + 0.5f - nx / 2 - x1) * temp + y1 + ny / 2;
            temp = (y2n - y1) / (x2n - x1);
            p2yn = (ix + 0.5f - nx / 2 - x1) * temp + y1 + ny / 2;
            if (p2y > p2yn)
            {ptmp = p2y; p2y = p2yn; p2yn = ptmp;}
            dst = p2yn - p2y;
            if (p2yn < 0.0f)
                continue;
            if (p2y >= (float)ny)
                continue;
            if (p2y < 0.0f)
                p2y = 0.0f;
            if (p2yn >= ny)
                p2yn = float(ny);
            dsp = MIN(p2yn, iy + 1) - MAX(iy, p2y); ds = dsp / dst;
            if (dsp < 0)
                continue;
            for (int ib = (int)floor(bs); ib < (int)ceil(be); ib++)
            {
                z2 = (bi + ib - 0.5f) * db;
                z2n = (bi + ib + 0.5f) * db;
                temp = (z2 - z1) / (x2m - x1);
                p2z = (ix + 0.5f - nx / 2 - x1) * temp + z1 + nz / 2;
                temp = (z2n - z1) / (x2m - x1);
                p2zn = (ix + 0.5f - nx / 2 - x1) * temp + z1 + nz / 2;
                if (p2z > p2zn)
                {ptmp = p2z; p2z = p2zn; p2zn = ptmp;}                            
                det = p2zn - p2z;
                if (p2zn < 0.0f)
                    continue;
                if (p2z >= (float)nz)
                    continue;
                if (p2z < 0.0f)
                    p2z = 0.0f;
                if (p2zn > nz)
                    p2zn = float(nz);
                //dt = MIN(p2zn, iz + 1) - MAX(iz, p2z); dt /= det;
                dtp = MIN(p2zn, iz + 1) - MAX(iz, p2z); dt = dtp / det;
                if (dtp < 0)
                    continue;
                img[id] += (proj[ib + (na - 1 - ia) * na] * ds * dt);
            }
        }                                    
    }
    else
    {
        yc = iy - ny / 2 + 0.5f;
        xc = ix - nx / 2;
        xcn = ix - nx / 2 + 1.0f;
        xcm = (xc + xcn) / 2;
        xc0 = cphi * xc + sphi * yc + SO;
        yc0 = -sphi * xc + cphi * yc;
        as = yc0 / xc0 * SD / da - ai + 0.5f;
        xc0 = cphi * xcn + sphi * yc + SO;
        yc0 = -sphi * xcn + cphi * yc;
        ae = yc0 / xc0 * SD / da - ai + 0.5f;
        if (as > ae)
        {atmp = as; as = ae; ae = atmp;}
        if (ae < 0.0f || as >= (float)na)
            return;              
        if (as < 0.0f)
            as = 0.0f;
        if (ae >= na)
            ae = float(na);
        zc = iz - nz / 2;
        zcn = iz - nz / 2 + 1.0f;
        xc0 = cphi * xcm + sphi * yc;
        yc0 = -sphi * xcm + cphi * yc;
        L = (float)sqrt((xc0 + SO) * (xc0 + SO) + yc0 * yc0);
        bs = (zc - z1) / L * SD / db - bi + 0.5f;
        be = (zcn - z1) / L * SD / db - bi + 0.5f;
        if (bs > be)
        {btmp = bs; bs = be; be = btmp;}
        if (be < 0.0f || bs >= (float)nb)
            return;
        if (bs < 0.0f)
            bs = 0.0f;
        if (be >= nb)
            be = float(nb);

        for (int ia = (int)floor(as); ia < (int)ceil(ae); ia++)
        {
            x20 = SD - SO;
            y20 = (ia + ai - 0.5f) * da;
            x2 = x20 * cphi - y20 * sphi;
            y2 = x20 * sphi + y20 * cphi;
            x20 = SD - SO;
            y20 = (ia + ai + 0.5f) * da;
            x2n = x20 * cphi - y20 * sphi;
            y2n = x20 * sphi + y20 * cphi;
            x2m = (x2 + x2n) / 2;
            y2m = (y2 + y2n) / 2;
            temp = (x2 - x1) / (y2 - y1);
            p2x = (iy + 0.5f - ny / 2 - y1) * temp + x1 + nx / 2;
            temp = (x2n - x1) / (y2n - y1);
            p2xn = (iy + 0.5f - ny / 2 - y1) * temp + x1 + nx / 2;
            if (p2x > p2xn)
            {ptmp = p2x; p2x = p2xn; p2xn = ptmp;}
            dst = p2xn - p2x;
            if (p2xn < 0.0f)
                continue;
            if (p2x >= (float)nx)
                continue;
            if (p2x < 0.0f)
                p2x = 0.0f;
            if (p2xn >= nx)
                p2xn = float(nx);
            dsp = MIN(p2xn, ix + 1) - MAX(ix, p2x); ds = dsp / dst;
            if (dsp < 0)
                continue;
            for (int ib = (int)floor(bs); ib < (int)ceil(be); ib++)
            {
                z2 = (bi + ib - 0.5f) * db;
                z2n = (bi + ib + 0.5f) * db;
                temp = (z2 - z1) / (y2m - y1);
                p2z = (iy + 0.5f - ny / 2 - y1) * temp + z1 + nz / 2;
                temp = (z2n - z1) / (y2m - y1);
                p2zn = (iy + 0.5f - ny / 2 - y1) * temp + z1 + nz / 2;
                if (p2z > p2zn)
                {ptmp = p2z; p2z = p2zn; p2zn = ptmp;}                            
                det = p2zn - p2z;
                if (p2zn < 0.0f)
                    continue;
                if (p2z >= (float)nz)
                    continue;
                if (p2z < 0.0f)
                    p2z = 0.0f;
                if (p2zn > nz)
                    p2zn = float(nz);
                //dt = MIN(p2zn, iz + 1) - MAX(iz, p2z); dt /= det;
                dtp = MIN(p2zn, iz + 1) - MAX(iz, p2z); dt = dtp / det;
                if (dtp < 0)
                    continue;
                img[id] += (proj[ib + (na - 1 - ia) * na] * ds * dt);
            }
        }       
    }
}