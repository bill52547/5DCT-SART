#include "dist_cuda_functions.h"
__host__ void kernel_back_dl(float *img, float *proj, float angle, float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz)
{
    float SI, DI, du, dv;
    size_t nu, nv, nt;
    SI = SO;
    DI = SD - SO;
    du = da;
    dv = db;
    nu = na; 
    nv = nb;
    nt = 1;
    float hh_sourcePhi[1];
    hh_sourcePhi[0] = angle;

    // Helical?
    float* hh_sourceZ;
    hh_sourceZ = NULL;
    // Image volume size
    size_t nVoxels = nx * ny * nz;

    // Calculate memory allocation sizes
    size_t nBytesVolume = nVoxels * sizeof(float);
    size_t nBytesProjections = nu * nv * nt * sizeof(float);
    size_t nBytesTimepointVector = nt * sizeof(float);
    size_t nBytesGeometry = 10 * sizeof(float);

    float *hh_geometry = new float [10];
    dist_back_project(img, proj, hh_sourcePhi, hh_sourceZ, &SI, &DI, &du, &dv, nu, nv, nx, ny, nz, nt, nBytesTimepointVector, hh_geometry, nBytesGeometry, nBytesVolume);
}
