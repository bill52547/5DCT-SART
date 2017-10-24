// past distance driven
#include <math.h>

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(x) ((x) > 0 ? (x) : -(x))
#define PI 3.141592653589793f
// #include "dist_cuda_functions.h"

__device__ void get_uv_ind_flat(int* uvInd, float u, float v, float du, float dv, float nu, float nv){

	// get index of detector element
	uvInd[0] = (int) floorf((u / du) + (nu/2) + 0.0f);
	uvInd[1] = (int) floorf((v / dv) + (nv/2) + 0.0f);

	//uvInd[0] = floorf((u / dv) + (nv/2) + 0.0f);
	//uvInd[1] = floorf((v / du) + (nu/2) + 0.0f);
}
// Find intersection of a line with detector plane
__device__ void get_intersection_line_plane(float* intersection, float* p1, float* p2, float* x0, float* n){

	// p1: first point on line
	// p2: second point on line
	// x0: point on plane
	// n: normal vector to plane (normalized or not)

	float r;
	float dot_plane;
	float dot_line;

	dot_plane = (n[0] * (x0[0] - p1[0])) + (n[1] * (x0[1] - p1[1])) + (n[2] * (x0[2] - p1[2]));
	dot_line = (n[0] * (p2[0] - p1[0])) + (n[1] * (p2[1] - p1[1])) + (n[2] * (p2[2] - p1[2]));

	r = dot_plane / dot_line;

	intersection[0] = p1[0] + r * (p2[0] - p1[0]);
	intersection[1] = p1[1] + r * (p2[1] - p1[1]);
	intersection[2] = p1[2] + r * (p2[2] - p1[2]);

}

// Convert from world x,y,z to detector plane u,v coordinates
 __device__ void xyz2uv(float*uv, float* x, float* x0, float* u1, float* v1){

	// returns distance along u1, and v1 from central_detector_array_position in units of mm (dx = dy = dz = 1 mm)
	float x1[3];

	x1[0] = x[0] - x0[0];
	x1[1] = x[1] - x0[1];
	x1[2] = x[2] - x0[2];

	uv[0] = (x1[0] * u1[0]) + (x1[1] * u1[1]) + (x1[2] * u1[2]);
	uv[1] = (x1[0] * v1[0]) + (x1[1] * v1[1]) + (x1[2] * v1[2]);
}
__device__ void get_uv_ranges(float* uMin, float* uMax, float* vMin, float* vMax, float* uv, float* dd_intersection, float* a1, float* a2, float* z1, float* z2, float* dd_sourcePosition, float* dd_centralDetectorPosition, float* u1, float* v1, float* dd_helical_detector_vector){


	float iu1, iu2, iu3, iu4;
	float iv1, iv2, iv3, iv4;
//	float iv1, iv2;
	float tmp;

	get_intersection_line_plane(dd_intersection, dd_sourcePosition, a1, dd_centralDetectorPosition, dd_helical_detector_vector);
//	xyz2uv(uv, dd_intersection, dd_helical_detector_vector, u1, v1);
	xyz2uv(uv, dd_intersection, dd_centralDetectorPosition, u1, v1);
	iu1 = uv[0];
	iv1 = uv[1];

	get_intersection_line_plane(dd_intersection, dd_sourcePosition, a2, dd_centralDetectorPosition, dd_helical_detector_vector);
	xyz2uv(uv, dd_intersection, dd_centralDetectorPosition, u1, v1);
	//xyz2uv(uv, dd_intersection, dd_helical_detector_vector, u1, v1);
	iu2 = uv[0];
	iv2 = uv[1];

	get_intersection_line_plane(dd_intersection, dd_sourcePosition, z1, dd_centralDetectorPosition, dd_helical_detector_vector);
	xyz2uv(uv, dd_intersection, dd_centralDetectorPosition, u1, v1);
	//xyz2uv(uv, dd_intersection, dd_helical_detector_vector, u1, v1);
	iu3 = uv[0];
	iv3 = uv[1];

	get_intersection_line_plane(dd_intersection, dd_sourcePosition, z2, dd_centralDetectorPosition, dd_helical_detector_vector);
	xyz2uv(uv, dd_intersection, dd_centralDetectorPosition, u1, v1);
	//xyz2uv(uv, dd_intersection, dd_helical_detector_vector, u1, v1);
	iu4 = uv[0];
	iv4 = uv[1];

	tmp = MIN(iu1,iu2);
	tmp = MIN(tmp,iu3);
	*uMin = MIN(tmp, iu4);

	tmp = MAX(iu1, iu2);
	tmp = MAX(tmp,iu3);
	*uMax = MAX(tmp, iu4);

	tmp = MIN(iv1,iv2);
	tmp = MIN(tmp,iv3);
	*vMin = MIN(tmp, iv4);

	tmp = MAX(iv1, iv2);
	tmp = MAX(tmp,iv3);
	*vMax = MAX(tmp, iv4);
}

__device__ void kernel_kernel(float* value, cudaTextureObject_t tex_proj, float angle, float SO, float SD, float nu, float nv, float du, float dv, float nx, float ny, float nz, int ix, int iy, int iz)
{
    value[0] = 0.0f;
    float dd_voxel[3];
	dd_voxel[0] = (float)ix - (nx/2) + 0.5f;
	dd_voxel[1] = (float)iy - (ny/2) + 0.5f;
	dd_voxel[2] = (float)iz - (nz/2) + 0.5f;

	// voxel boundary coordinates
	float a1[3];
	float a2[3];
	float z1[3];
	float z2[3];

	//intermediate variables
	float uMin, uMax;
	float vMin, vMax;
	float dd_uv[2];

	float uMinInd, uMaxInd;
	int vMinInd, vMaxInd;

	float uBound1, uBound2;
	float vBound1, vBound2;


	float wu, wv, w;

	float dd_centralDetectorPosition_rotated[3];
	float dd_sourcePosition_rotated[3];

	float dd_intersection[3];

	float u1_rotated[3];
	float v1_rotated[3];
	int dd_uvInd[2];

	// Helical scan?
	float dd_helical_detector_vector[3];
	// Normalize? 

	float tmp;

	//Loop counters
	int iV, iU;

	// Rotate to phi = 0
	float sphi = 0.0f;
	float cphi = 1.0f;
	float* dd_cos = &cphi;
	float* dd_sin = &sphi;
    __sincosf(angle, dd_sin, dd_cos); 
	dd_centralDetectorPosition_rotated[0] = (SD - SO) * - 1;
	dd_centralDetectorPosition_rotated[1] = 0;
	dd_centralDetectorPosition_rotated[2] = 0;

	dd_sourcePosition_rotated[0] = SO;
	dd_sourcePosition_rotated[1] = 0;
	dd_sourcePosition_rotated[2] = 0;

	u1_rotated[0] = 0.0f;
	u1_rotated[1] = 0.0f;
	u1_rotated[2] = 1.0f;

	v1_rotated[0] = 0.0f;
	v1_rotated[1] = 1.0f;
	v1_rotated[2] = 0.0f;

	dd_helical_detector_vector[0] = dd_centralDetectorPosition_rotated[0];
	dd_helical_detector_vector[1] = dd_centralDetectorPosition_rotated[1];
	dd_helical_detector_vector[2] = 0;


	// cos and sin of projection angle
    // sphi = (float)sinf(angle);
    // cphi = (float)cosf(angle);

	// get rotated coordinates of voxel edges:
	//left
	a1[0] = dd_voxel[0] * cphi + dd_voxel[1] * sphi;
	a1[1] = dd_voxel[0] * -1 * sphi + dd_voxel[1] * cphi;
	a1[2] = dd_voxel[2];

	a1[1] = a1[1] - 0.5f;
	a1[0] = a1[0] - 0.5f;

	// right
	a2[0] = dd_voxel[0] * cphi + dd_voxel[1] * sphi;
	a2[1] = dd_voxel[0] * -1 * sphi + dd_voxel[1] * cphi;
	a2[2] = dd_voxel[2];

	a2[1] = a2[1] + 0.5f;
	a2[0] = a2[0] + 0.5f;

	//lower
	z1[0] = (dd_voxel[0] * cphi) + (dd_voxel[1] * sphi);
	z1[1] = -1 * (dd_voxel[0] * sphi) + (dd_voxel[1] * cphi);
	z1[2] = dd_voxel[2] - 0.5f;

	//upper
	z2[0] = (dd_voxel[0] * cphi) + (dd_voxel[1] * sphi);
	z2[1] = -1 * (dd_voxel[0] * sphi) + (dd_voxel[1] * cphi);
	z2[2] = dd_voxel[2] + 0.5f;

      // get intersection of ray from source through voxel edges and detector plane to find shadow region of this voxel
	get_uv_ranges(&uMin, &uMax, &vMin, &vMax, dd_uv, dd_intersection, a1, a2, z1, z2, dd_sourcePosition_rotated, dd_centralDetectorPosition_rotated, u1_rotated, v1_rotated,dd_helical_detector_vector);

      // convert from distance in voxel spacing (1mm) from detector array center in u and v to element indices (1,2,etc..)
	get_uv_ind_flat(dd_uvInd, uMin, vMin, du, dv, nu, nv);
	uMinInd = dd_uvInd[0];
	vMinInd = dd_uvInd[1];
	
	get_uv_ind_flat(dd_uvInd, uMax, vMax, du, dv, nu, nv);
	uMaxInd = dd_uvInd[0];
	vMaxInd = dd_uvInd[1];
	
	if(uMin > uMax){
		tmp = uMin;
		uMin = uMax;
		uMax = tmp;}
	
	if(vMin > vMax){
		tmp = vMin;
		vMin = vMax;
		vMax = tmp;}
	
	// loop over detectors in the shadow region of this voxel
	for (iV = MAX(0,vMinInd); iV <= MIN(vMaxInd, ((int)nv-1)); iV++){
	for (iU = MAX(0,uMinInd); iU <= MIN(uMaxInd, ((int)nu-1)); iU++){


		vBound1 = ((float)iV - 0.5f - (nv/2) + 0.5f) * dv;
		vBound2 = ((float)iV + 0.5f - (nv/2) + 0.5f) * dv;

		// v weight
		wv = 0;
			if ( (vBound1 < vMin) && (vMax >= vBound2) ){
			wv = (vBound2 - vMin) / dv;
			}

			else if ( (vBound1 < vMin) && (vMax < vBound2) ){
			wv = (vMax - vMin) / dv;
			}

			else if ( (vBound2 > vMax) && (vMin <= vBound1) ) {
			wv = (vMax - vBound1) / dv;
			}

			else{
			wv = 1;
			}

		uBound1 = (iU - 0.5f - (nu/2) + 0.5f) * du;
		uBound2 = (iU + 0.5f - (nu/2) + 0.5f) * du;

		// u weight
		wu = 0;

			if ( (uBound1 < uMin) && (uMax > uBound2) ){
			wu = (uBound2 - uMin) / du;
			}

			else if ( (uBound1 < uMin) && (uMax <= uBound2) ){

			//wu = 1;
			wu = (uMax - uMin) / du;
			}

			else if ( (uBound2 > uMax) && (uMin <= uBound1) ) {
			wu = (uMax - uBound1) / du;
			}

			else{
			wu = 1;
			}

		w = wu * wv;

		value[0] += w * tex3D<float>(tex_proj,(iU + 0.5f), (iV + 0.5f), 0.5f);

// end loop over detectors
	}
	}
}

__global__ void kernel(float *img, cudaTextureObject_t tex_proj, float angle, float SO, float SD, float nu, float nv, float du, float dv, float nx, float ny, float nz){
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    img[id] = 0.0f;
    kernel_kernel(&img[id], tex_proj, angle, SO, SD, nu, nv, du, dv, nx, ny, nz, ix, iy ,iz);
    // world coordinates for this voxel
	
}


__host__ void kernel_backprojection(float *img, float *proj, float angle, float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    struct cudaExtent extent = make_cudaExtent(nb, na, 1);
    cudaArray *array_proj;
    cudaMalloc3DArray(&array_proj, &channelDesc, extent);
    cudaMemcpy3DParms copyParams = {0};
    cudaPitchedPtr dp_proj = make_cudaPitchedPtr((void*) proj, nb * sizeof(float), nb, na);
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

    const dim3 gridSize_singleProj((nx + 16 - 1) / 16, (ny + 16 - 1) / 16, (nz + 3) / 4);
    const dim3 blockSize(16, 16, 4);
    kernel<<<gridSize_singleProj, blockSize>>>(img, tex_proj, angle, SO, SD, nb, na, da, db, nx, ny, nz);
    cudaDeviceSynchronize();

    cudaFreeArray(array_proj);
    cudaDestroyTextureObject(tex_proj);

}

