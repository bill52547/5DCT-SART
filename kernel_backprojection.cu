#include "kernel_backprojection.h"

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
    // cudaTextureObject_t tex_proj = host_create_texture_object(d_proj, nb, na, 1);
    cudaCreateTextureObject(&tex_proj, &resDesc, &texDesc, NULL);

    const dim3 gridSize_img((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (nz + BLOCKDEPTH - 1) / BLOCKDEPTH);
    const dim3 blockSize(BLOCKWIDTH, BLOCKHEIGHT, BLOCKDEPTH);
	// mexPrintf("angle = %f.\n", angle);
	// mexPrintf("SO = %f.\n", SO);
	// mexPrintf("SD = %f.\n", SD);
	// mexPrintf("na = %d.\n", na);
	// mexPrintf("nb = %d.\n", nb);
	// mexPrintf("da = %f.\n", da);
	// mexPrintf("db = %f.\n", db);
	// mexPrintf("ai = %f.\n", ai);
	// mexPrintf("bi = %f.\n", bi);
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
	// angle += 3.141592653589793;

    img[id] = 0.0f;
	// float sphi = __sinf(angle);
	// float cphi = __cosf(angle);
	float sphi = __sinf(angle);
	float cphi = __cosf(angle);
	// float dd_voxel[3];
	float xc, yc, zc;
	xc = (float)ix - nx / 2 + 0.5f;
	yc = (float)iy - ny / 2 + 0.5f;
	zc = (float)iz - nz / 2 + 0.5f;

	// voxel boundary coordinates
	float xl, yl, zl, xr, yr, zr, xt, yt, zt, xb, yb, zb;
	xl = xc * cphi + yc * sphi;// - 0.5f;
    yl = -xc * sphi + yc * cphi - 0.5f;
    xr = xc * cphi + yc * sphi;// + 0.5f;
    yr = -xc * sphi + yc * cphi + 0.5f;
    zl = zc; zr = zc;
    xt = xc * cphi + yc * sphi;
    yt = -xc * sphi + yc * cphi;
    zt = zc + 0.5f;
    xb = xc * cphi + yc * sphi;
    yb = -xc * sphi + yc * cphi;
    zb = zc - 0.5f;

	// the coordinates of source and detector plane here are after rotation
	float ratio, al, bl, ar, br, at, bt, ab, bb, a_max, a_min, b_max, b_min;
	// calculate a value for each boundary coordinates
	

	// the a and b here are all absolute positions from isocenter, which are on detector planes
	ratio = SD / (xl + SO);
	al = ratio * yl;
	bl = ratio * zl;
	ratio = SD / (xr + SO);
	ar = ratio * yr;
	br = ratio * zr;
	ratio = SD / (xt + SO);
	at = ratio * yt;
	bt = ratio * zt;
	ratio = SD / (xb + SO);
	ab = ratio * yb;
	bb = ratio * zb;

	// get the max and min values of all boundary projectors of voxel boundaries on detector plane
	a_max = MAX4(al ,ar, at, ab);
	a_min = MIN4(al ,ar, at, ab);
	b_max = MAX4(bl ,br, bt, bb);
	b_min = MIN4(bl ,br, bt, bb);

	// the related positions on detector plane from start points
	a_max = a_max / da - ai + 0.5f; //  now they are the detector coordinates
	a_min = a_min / da - ai + 0.5f;
	b_max = b_max / db - bi + 0.5f;
	b_min = b_min / db - bi + 0.5f;
	int a_ind_max = (int)floorf(a_max); 	
	int a_ind_min = (int)floorf(a_min); 
	int b_ind_max = (int)floorf(b_max); 
	int b_ind_min = (int)floorf(b_min); 
	
	// int a_ind_max = (int)floorf(a_max / da - ai);
	// int a_ind_min = (int)floorf(a_min / da - ai);
	// int b_ind_max = (int)floorf(b_max / db - bi);
	// int b_ind_min = (int)floorf(b_min / db - bi);

	float bin_bound_1, bin_bound_2, wa, wb;
	for (int ia = MAX(0, a_ind_min); ia <= MIN((na - 1), a_ind_max); ia ++){
		// bin_bound_1 = ((float)ia + ai) * da;
		// bin_bound_2 = ((float)ia + ai + 1.0f) * da;
		bin_bound_1 = ia + 0.0f;
		bin_bound_2 = ia + 1.0f;
		
		wa = MIN(bin_bound_2, a_max) - MAX(bin_bound_1, a_min);// wa /= da;

		for (int ib = MAX(0, b_ind_min); ib <= MIN((nb - 1), b_ind_max); ib ++){
			// bin_bound_1 = ((float)ib + bi) * db;
			// bin_bound_2 = ((float)ib + bi + 1.0f) * db;
			bin_bound_1 = ib + 0.0f;
			bin_bound_2 = ib + 1.0f;
			wb = MIN(bin_bound_2, b_max) - MAX(bin_bound_1, b_min);// wb /= db;
			// img[id] = tex3D<float>(tex_proj, (ib + 0.5f), (ia + 0.5f), 0.5f);
			// return;		

			img[id] += wa * wb * tex3D<float>(tex_proj, (ia + 0.5f), (ib + 0.5f), 0.5f);
		}		
	}
}