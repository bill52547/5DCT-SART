__global__ void kernel_projection(float *proj, float *img, float angle, float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz);
// #define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
// #define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))