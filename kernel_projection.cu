#include <math.h>
#define ABS(x) ((x) > 0 ? (x) : - (x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

__global__ void kernel_projection(float *proj, float *img, float angle, float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz){
    int ia = 16 * blockIdx.x + threadIdx.x;
    int ib = 16 * blockIdx.y + threadIdx.y;
    if (ia >= na || ib >= nb)
        return;
    int id = ia + ib * na;
    proj[id] = 0.0f;
    angle += 3.141592653589793;
    float x1, y1, z1, x2, y2, z2, x20, y20, cphi, sphi;
    cphi = (float)cosf(angle);
    sphi = (float)sinf(angle);
    x1 = -SO * cphi;
    y1 = -SO * sphi;
    z1 = 0.0f;
    x20 = SD - SO;
    y20 = (ia + ai) * da; // locate the detector cell center before any rotation
    x2 = x20 * cphi - y20 * sphi;
    y2 = x20 * sphi + y20 * cphi;
    z2 = (ib + bi) * db;
    float x21, y21, z21; // offset between source and detector center
    x21 = x2 - x1;
    y21 = y2 - y1;
    z21 = z2 - z1;

    // y - z plane, where ABS(x21) > ABS(y21)
    if (ABS(x21) > ABS(y21)){
        float yi1, yi2, ky1, ky2, zi1, zi2, kz1, kz2;
        int Yi1, Yi2, Zi1, Zi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int ix = 0; ix < nx; ix++){
            // calculate y indices of intersecting voxel candidates
            ky1 = (y21 - da / 2 * cphi) / (x21 + da / 2 * sphi);
            yi1 = ky1 * ((float)ix + 0.5f - x1 - nx / 2) + y1 + ny / 2;
            Yi1 = (int)floor(yi1); // lower boundary of related voxels at y-axis
            ky2 = (y21 + da / 2 * cphi) / (x21 - da / 2 * sphi);
            yi2 = ky2 * ((float)ix + 0.5f - x1 - nx / 2) + y1 + ny / 2;
            Yi2 = (int)floor(yi2); // upper boundary of related voxels at y-axis
            // if (Yi1 < 0)
            //     Yi1 = 0;
            // if (Yi2 >= ny)
            //     Yi2 = ny - 1;

            // calculate z indices of intersecting voxel candidates
            kz1 = (z21 - db / 2) / x21;
            zi1 = kz1 * ((float)ix + da * 0.5f - x1 - nx / 2) + z1 + nz / 2;
            Zi1 = (int)floor(zi1); // lower boundary of related voxels at y-axis
            kz2 = (z21 + db / 2) / x21;
            zi2 = kz2 * ((float)ix + da * 0.5f - x1 - nx / 2) + z1 + nz / 2;
            Zi2 = (int)floor(zi2); // upper boundary of related voxels at y-axis
            // if (Zi1 < 0)
            //     Zi1 = 0;
            // if (Zi2 >= nz)
            //     Zi2 = nz - 1;

            // calculate contribution of a voxel to the projection value
            int iy, iz;
            float wy1, wy2, wz1, wz2;
            wy1 = (MAX(Yi1, Yi2) - yi1) / (yi2 - yi1); wy2 = 1 - wy1;
            wz1 = (MAX(Zi1, Zi2) - zi1) / (zi2 - zi1); wz2 = 1 - wz1;

            // Yi1 == Yi2 && Zi1 == Zi2
            if (Yi1 == Yi2 && Zi1 == Zi2)
            {
                iy = Yi1; iz = Zi1; 
                if (iy < ny && iy >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * 1.0f;
                continue;
            }
            // Yi1 != Yi2 && Zi1 == Zi2
            if (Yi1 != Yi2 && Zi1 == Zi2)
            {
                iy = Yi1; iz = Zi1; 
                if (iy < ny && iy >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wy1;
                iy = Yi2; iz = Zi1; 
                if (iy < ny && iy >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wy2;
                continue;                
            }
            // Yi1 == Yi2 && Zi1 != Zi2
            if (Yi1 == Yi2 && Zi1 != Zi2)
            {
                iy = Yi1; iz = Zi1; 
                if (iy < ny && iy >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wz1;
                iy = Yi1; iz = Zi2; 
                if (iy < ny && iy >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wz2;
                continue;                
            }
            // Yi1 != Yi2 && Zi1 != Zi2
            if (Yi1 != Yi2 && Zi1 != Zi2)
            {
                iy = Yi1; iz = Zi1; 
                if (iy < ny && iy >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wy1 * wz1;
                iy = Yi1; iz = Zi2; 
                if (iy < ny && iy >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wy1 * wz2;
                iy = Yi2; iz = Zi1; 
                if (iy < ny && iy >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wy2 * wz1;
                iy = Yi2; iz = Zi2; 
                if (iy < ny && iy >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wy2 * wz2;
                continue;                
            }
        }
    }
    // x - z plane, where ABS(x21) <= ABS(y21)    
    else{
    float xi1, xi2, kx1, kx2, zi1, zi2, kz1, kz2;
        int Xi1, Xi2, Zi1, Zi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int iy = 0; iy < ny; iy++){
            // calculate y indices of intersecting voxel candidates
            kx1 = (x21 - da / 2 * sphi) / (y21 + da / 2 * cphi);
            xi1 = kx1 * ((float)iy + 0.5f - y1 - ny / 2) + x1 + nx / 2;
            Xi1 = (int)floor(xi1); // lower boundary of related voxels at y-axis
            kx2 = (x21 + da / 2 * sphi) / (y21 - da / 2 * cphi);
            xi2 = kx2 * ((float)iy + 0.5f - y1 - ny / 2) + x1 + nx / 2;
            Xi2 = (int)floor(xi2); // upper boundary of related voxels at y-axis
            // if (Xi1 < 0)
            //     Xi1 = 0;
            // if (Xi2 >= ny)
            //     Xi2 = ny - 1;

            // calculate z indices of intersecting voxel candidates
            kz1 = (z21 - db / 2) / y21;
            zi1 = kz1 * ((float)iy + da * 0.5f - y1 - ny / 2) + z1 + nz / 2;
            Zi1 = (int)floor(zi1); // lower boundary of related voxels at y-axis
            kz2 = (z21 + db / 2) / y21;
            zi2 = kz2 * ((float)iy + da * 0.5f - y1 - ny / 2) + z1 + nz / 2;
            Zi2 = (int)floor(zi2); // upper boundary of related voxels at y-axis
            // if (Zi1 < 0)
            //     Zi1 = 0;
            // if (Zi2 >= nz)
            //     Zi2 = nz - 1;

            // calculate contribution of a voxel to the projection value
            int ix, iz;
            float wx1, wx2, wz1, wz2;
            wx1 = (MAX(Xi1, Xi2) - xi1) / (xi2 - xi1); wx2 = 1 - wx1;
            wz1 = (MAX(Zi1, Zi2) - zi1) / (zi2 - zi1); wz2 = 1 - wz1;

            // Xi1 == Xi2 && Zi1 == Zi2
            if (Xi1 == Xi2 && Zi1 == Zi2)
            {
                ix = Xi1; iz = Zi1; 
                if (ix < nx && ix >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * 1.0f;
                continue;
            }
            // Xi1 != Xi2 && Zi1 == Zi2
            if (Xi1 != Xi2 && Zi1 == Zi2)
            {
                ix = Xi1; iz = Zi1; 
                if (ix < nx && ix >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wx1;
                ix = Xi2; iz = Zi1; 
                if (ix < nx && ix >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wx2;
                continue;                
            }
            // Xi1 == Xi2 && Zi1 != Zi2
            if (Xi1 == Xi2 && Zi1 != Zi2)
            {
                ix = Xi1; iz = Zi1; 
                if (ix < nx && ix >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wz1;
                ix = Xi1; iz = Zi2; 
                if (ix < nx && ix >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wz2;
                continue;                
            }
            // Xi1 != Xi2 && Zi1 != Zi2
            if (Xi1 != Xi2 && Zi1 != Zi2)
            {
                ix = Xi1; iz = Zi1; 
                if (ix < nx && ix >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wx1 * wz1;
                ix = Xi1; iz = Zi2; 
                if (ix < nx && ix >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wx1 * wz2;
                ix = Xi2; iz = Zi1; 
                if (ix < nx && ix >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wx2 * wz1;
                ix = Xi2; iz = Zi2; 
                if (ix < nx && ix >= 0 && iz < nz && iz >= 0)
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wx2 * wz2;
                continue;                
            }
        }
    }

        

    
}