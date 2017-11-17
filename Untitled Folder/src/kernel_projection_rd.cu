#include "kernel_projection_rd.cuh"
__host__ void kernel_projection(float *d_proj, float *d_img, float angle, float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz){
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    struct cudaExtent extent = make_cudaExtent(nx, ny, nz);
    cudaArray *array_img;
    cudaMalloc3DArray(&array_img, &channelDesc, extent);
    cudaMemcpy3DParms copyParams = {0};
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) d_img, nx * sizeof(float), nx, ny);
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_img;
    copyParams.dstArray = array_img;
    cudaMemcpy3D(&copyParams);

    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    resDesc.res.array.array = array_img;
	cudaTextureObject_t tex_img = 0;
    cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);

    const dim3 gridSize_img((na + BLOCKWIDTH - 1) / BLOCKWIDTH, (nb + BLOCKHEIGHT - 1) / BLOCKHEIGHT, 1);
    const dim3 blockSize(BLOCKWIDTH, BLOCKHEIGHT, BLOCKDEPTH);

    kernel<<<gridSize_img, blockSize>>>(d_proj, tex_img, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
    cudaDeviceSynchronize();

    cudaFreeArray(array_img);
    cudaDestroyTextureObject(tex_img);
}
__global__ void kernel(float *proj, cudaTextureObject_t tex_img, float angle, float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz){
    int ia = BLOCKWIDTH * blockIdx.x + threadIdx.x;
    int ib = BLOCKHEIGHT * blockIdx.y + threadIdx.y;
    if (ia >= na || ib >= nb)
        return;
    int id = ia + ib * na;
    proj[id] = 0.0f;
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

    float dl = 0.001f, maxL, L;
    L = (float)sqrt(x21 * x21 + y21 * y21 + z21 * z21);
    maxL = L / dl;
    float xc, yc, zc, alpha;
    for (int i = 0; i < maxL; i++)
    {
        alpha = i * dl / L;
        xc = x1 + alpha * x21 + nx / 2;
        yc = y1 + alpha * y21 + ny / 2;
        zc = z1 + alpha * z21 + nz / 2;
        proj[id] += tex3D<float>(tex_img, xc, yc, zc) * dl;
    }
}