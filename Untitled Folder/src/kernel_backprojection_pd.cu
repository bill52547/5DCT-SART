#include "kernel_backprojection.cuh"

__host__ void kernel_backprojection(float *d_img, float *d_proj, float angle,float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    struct cudaExtent extent = make_cudaExtent(na, nb, 1);
    cudaArray *array_proj;
    cudaMalloc3DArray(&array_proj, &channelDesc, extent);
    cudaMemcpy3DParms copyParams = {0};
    cudaPitchedPtr dp_proj = make_cudaPitchedPtr((void*) d_proj, na * sizeof(float), na, nb);
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_proj;
    copyParams.dstArray = array_proj;
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
    resDesc.res.array.array = array_proj;
	cudaTextureObject_t tex_proj = 0;
    cudaCreateTextureObject(&tex_proj, &resDesc, &texDesc, NULL);

    const dim3 gridSize_img((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (nz + BLOCKDEPTH - 1) / BLOCKDEPTH);
    const dim3 blockSize(BLOCKWIDTH, BLOCKHEIGHT, BLOCKDEPTH);

    kernel<<<gridSize_img, blockSize>>>(d_img, tex_proj, angle, SO, SD, na, nb, da, db, ai, bi, nx, ny, nz);
    cudaDeviceSynchronize();

    cudaFreeArray(array_proj);
    cudaDestroyTextureObject(tex_proj);
}

__global__ void kernel(float *img, cudaTextureObject_t tex_proj, float angle, float SO, float SD, int na, int nb, float da, float db, float ai, float bi, int nx, int ny, int nz){
    int ix = BLOCKWIDTH * blockIdx.x + threadIdx.x;
    int iy = BLOCKHEIGHT * blockIdx.y + threadIdx.y;
    int iz = BLOCKDEPTH * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;

    int id = ix + iy * nx + iz * nx * ny;

    img[id] = 0.0f;

	float sphi = __sinf(angle);
	float cphi = __cosf(angle);
	// float dd_voxel[3];
	float xc, yc, zc, xc0, yc0;
	xc0 = (float)ix - nx / 2 + 0.5f;
	yc0 = (float)iy - ny / 2 + 0.5f;
	zc = (float)iz - nz / 2 + 0.5f;
	xc = xc0 * cphi + yc0 * sphi;
	yc = -xc0 * sphi + yc0 * cphi;

	float x1, y1, z1;
	x1 = -SO;
	y1 = 0;
	z1 = 0;
	
	float x2, y2, z2;
	x2 = SD - SO;
	y2 = (x2 - x1) / (xc - x1) * (yc - y1) + y1;
	z2 = (x2 - x1) / (xc - x1) * (zc - z1) + z1;

	float a, b;
	a = y2 - ai + 0.5f;
	b = z2 - bi + 0.5f;
	
	img[id] += tex3D<float>(tex_proj, a, b, 0.5f);
}