#include "dist_cuda_functions.h"

// Cases for slicing volume into planes
static const int XZ_PLANE = 0;
static const int YZ_PLANE = 1;

// Constant memory for geometry constants
__constant__ float hdc_geometry[N_GEOMETRY_CONSTANTS];

// Get coordinates of the central point of detector array (same for planar and arc)
 __device__ void get_central_detector_array_position(float* dd_centralDetectorPosition, float* dd_sourcePosition, float cosPhi, float sinPhi, float DI, float sourceZ){

	dd_centralDetectorPosition[0] = cosPhi * DI * - 1;
	dd_centralDetectorPosition[1] = sinPhi * DI * - 1;
	dd_centralDetectorPosition[2] = sourceZ;
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

// Go from u,v coordinate to detector index, flat panel
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

// ** Flat detector array
// Get u1 and v1, 2 orthonormal basis vectors describing the detector plane
 __device__ void get_detector_basis_vectors_flat(float* dd_sourcePosition, float* u1, float* v1){

	// one axis of the detector plane will always be parallel to the Z axis.  use this as one basis vector
	u1[0] = 0;
	u1[1] = 0;
	u1[2] = 1;
//	u1[2] = -1;

	// Cross product to find a vector orthogonal to the z axis and the source position
	v1[0] = (dd_sourcePosition[1] * u1[2]) - (dd_sourcePosition[2] * u1[1]);
	v1[1] = (dd_sourcePosition[2] * u1[0]) - (dd_sourcePosition[0] * u1[2]);
	v1[2] = (dd_sourcePosition[0] * u1[1]) - (dd_sourcePosition[1] * u1[0]);

	// Normalize v1

	float vNorm = (v1[0] * v1[0]) + (v1[1] * v1[1]) + (v1[2] * v1[2]);
	vNorm = sqrt(vNorm);

	v1[0] = v1[0] / vNorm;
	v1[1] = v1[1] / vNorm;
	v1[2] = v1[2] / vNorm;
	v1[0] = v1[0] * -1;
	v1[1] = v1[1] * -1;
	v1[2] = v1[2] * -1;
 }

// Position of this thread's detector element
 __device__ void get_detector_position_flat(float* dd_detectorPosition, float* dd_centralDetectorPosition, float* u1, float* v1, unsigned int u, unsigned int v, float nu, float nv, float du, float dv){

	// go from u,v indices to u, v coordinates in detector plane
	float uu = (u - (nu/2) + 0.5f) * du;
	float vv = (v - (nv/2) + 0.5f) * dv;

	dd_detectorPosition[0] = uu * u1[0] + (vv * v1[0]);
	dd_detectorPosition[1] = uu * u1[1] + (vv * v1[1]);
	dd_detectorPosition[2] = uu * u1[2] + (vv * v1[2]);

	dd_detectorPosition[0] = dd_detectorPosition[0] + dd_centralDetectorPosition[0];
	dd_detectorPosition[1] = dd_detectorPosition[1] + dd_centralDetectorPosition[1];
	dd_detectorPosition[2] = dd_detectorPosition[2] + dd_centralDetectorPosition[2];

}

// Boundaries of this thread's detector element
__device__ void get_detector_boundary_positions_flat(float* dd_uF, float* dd_uL, float* dd_vF, float* dd_vL, float* dd_detectorPosition, float* dd_v1, float du, float dv, float cosPhi, float sinPhi){

	dd_uF[0] = dd_detectorPosition[0];
	dd_uF[1] = dd_detectorPosition[1];
	dd_uF[2] = dd_detectorPosition[2] - (du/2);

	dd_uL[0] = dd_detectorPosition[0];
	dd_uL[1] = dd_detectorPosition[1];
	dd_uL[2] = dd_detectorPosition[2] + (du/2);

	dd_vF[0] = dd_detectorPosition[0] - ((dv/2) * dd_v1[0]);
	dd_vF[1] = dd_detectorPosition[1] - ((dv/2) * dd_v1[1]);
	//dd_vF[0] = dd_detectorPosition[0] + ((dv/2) * sinPhi);
	//dd_vF[1] = dd_detectorPosition[1] - ((dv/2) * cosPhi);
	dd_vF[2] = dd_detectorPosition[2];

	dd_vL[0] = dd_detectorPosition[0] + ((dv/2) * dd_v1[0]);
	dd_vL[1] = dd_detectorPosition[1] + ((dv/2) * dd_v1[1]);
	//dd_vL[0] = dd_detectorPosition[0] - ((dv/2) * sinPhi);
	//dd_vL[1] = dd_detectorPosition[1] + ((dv/2) * cosPhi);
	dd_vL[2] = dd_detectorPosition[2];


}

// ** Source position
__device__ void get_source_position(float* dd_sourcePosition, float cosPhi, float sinPhi, float SI, float sourceZ){

	dd_sourcePosition[0] = cosPhi * SI;
	dd_sourcePosition[1] = sinPhi * SI;
	dd_sourcePosition[2] = sourceZ;
}

// Decide whether to project detector and voxel boundaries onto XZ or YZ plane
 __device__ int get_common_plane(float* dd_sourcePosition, float* dd_detectorPosition){

	// Compute slope of connecting line between center of detector cell and source focal point
	// and decide which plane to divide the 3D image

	float ddx = dd_detectorPosition[0] - dd_sourcePosition[0];
	float ddy = dd_detectorPosition[1] - dd_sourcePosition[1];

	ddx = ABS(ddx);
	ddy = ABS(ddy);

	int commonPlane = -1;

	if( ddx <= ddy){
		commonPlane = XZ_PLANE;
	}
	else{
		commonPlane = YZ_PLANE;
	}

	return commonPlane;
}

 __device__ float get_slab_intersection_length(float* dd_sourcePosition, float* dd_detectorPosition, int commonPlane, int u, float du, float nu){


	 // a = atan(y/SD);
	 // b = atan(z/SD);

	// Get line from source to detector element center
	float lx = dd_sourcePosition[0] - dd_detectorPosition[0]; 
	float ly = dd_sourcePosition[1] - dd_detectorPosition[1]; 
	float lz = dd_sourcePosition[2] - dd_detectorPosition[2]; 

	// Norm of line (distance)
	float lNorm = (lx * lx) + (ly * ly) + (lz * lz);
	lNorm = sqrtf(lNorm);

	float vx;
	float vy;
	float vz;
	

	// Determine which axis to find angle between
	switch (commonPlane) {

		case YZ_PLANE:

		vx = 1;
		vy = 0;
		vz = 0;
		break;


		case XZ_PLANE:

		vx = 0;
		vy = 1;
		vz = 0;
		break;
	}


	float alpha, gamma;
	float cx, cy, cz, cNorm;
	float dot;

	// Cross product between line from detector element to source and v axis of detector
	cx = (ly * vz) - (lz * vy);
	cy = (lz * vx) - (lx * vz);
	cz = (lx * vy) - (ly * vx);

	// Norm of cross product
	cNorm = (cx * cx) + (cy * cy) + (cz * cz);
	cNorm = sqrtf(cNorm);

	// Dot product between line from detector element to source and v axis of detector
	dot = (lx * vx) + (ly * vy) + (lz * vz);
	alpha = atan2f(cNorm, dot);

	// Get out-of-plane angle gamma
	float uu;
	//uu = u - (nu/2);
	uu = u - (nu/2) + 0.5f;
	uu = uu * du;
	uu = ABS(uu);
	gamma = asinf(uu/lNorm);

	float slabIntersectionLength = dx / (__cosf(alpha) * __cosf(gamma));
	slabIntersectionLength = ABS(slabIntersectionLength);

	return slabIntersectionLength;
}

// Calculate dx/dy and dz/dy (for XZ plane) or dy/dx and dz/dx ( for YZ plane)
 __device__ void get_slopes(float* ka1, float* ka2, float* kz1, float* kz2, float* dd_uF, float* dd_uL, float* dd_vF, float* dd_vL, float* dd_sourcePosition, int commonPlane){

	switch (commonPlane) {
		case YZ_PLANE:

			ka1[0] = (dd_vF[1] - dd_sourcePosition[1]) / (dd_vF[0] - dd_sourcePosition[0]);
			ka2[0] = (dd_vL[1] - dd_sourcePosition[1]) / (dd_vL[0] - dd_sourcePosition[0]);

			kz1[0] = (dd_uF[2] - dd_sourcePosition[2]) / (dd_uF[0] - dd_sourcePosition[0]);
			kz2[0] = (dd_uL[2] - dd_sourcePosition[2]) / (dd_uL[0] - dd_sourcePosition[0]);

			break;

		case XZ_PLANE:

			ka1[0] = (dd_vF[0] - dd_sourcePosition[0]) / (dd_vF[1] - dd_sourcePosition[1]);
			ka2[0] = (dd_vL[0] - dd_sourcePosition[0]) / (dd_vL[1] - dd_sourcePosition[1]);

			kz1[0] = (dd_uF[2] - dd_sourcePosition[2]) / (dd_uF[1] - dd_sourcePosition[1]);
			kz2[0] = (dd_uL[2] - dd_sourcePosition[2]) / (dd_uL[1] - dd_sourcePosition[1]);

			break;
	}
}


// Find the voxels that are viewed by this detector and their projected boundaries on the common plane
 __device__ void get_intersecting_voxels(float* A1, float* A2, float* a1, float* a2, float* Z1, float* Z2, float* z1, float* z2, float ka1, float ka2, float kz1, float kz2, float* dd_sourcePosition, float nx, float ny, float nz, float du, float dv, int commonPlane, int iSlice){
	
	switch (commonPlane) {
		
		case YZ_PLANE:

		//*a1 = ka1 * (iSlice - (nx/2) + 0.5f + (dy/2) - dd_sourcePosition[0]) + dd_sourcePosition[1] + (ny/2) - 0.0f;
		//*a2 = ka2 * (iSlice - (nx/2) + 0.5f + (dy/2) - dd_sourcePosition[0]) + dd_sourcePosition[1] + (ny/2) - 0.0f;
		*a1 = ka1 * (iSlice - (nx/2) +0.5f - dd_sourcePosition[0]) + dd_sourcePosition[1] + (ny/2) - 0.0f;
		*a2 = ka2 * (iSlice - (nx/2) +0.5f - dd_sourcePosition[0]) + dd_sourcePosition[1] + (ny/2) - 0.0f;


		//*z1 = ((kz1 * (iSlice - (nx/2) + 0.5f + (dz/2) - dd_sourcePosition[0]) + dd_sourcePosition[2]) / dz ) + (nz/2) - 0.0f;
		//*z2 = ((kz2 * (iSlice - (nx/2) + 0.5f + (dz/2) - dd_sourcePosition[0]) + dd_sourcePosition[2]) / dz ) + (nz/2) - 0.0f;
		*z1 = ((kz1 * (iSlice - (nx/2) +0.5f - dd_sourcePosition[0]) + dd_sourcePosition[2]) / dz ) + (nz/2) - 0.0f;
		*z2 = ((kz2 * (iSlice - (nx/2) +0.5f - dd_sourcePosition[0]) + dd_sourcePosition[2]) / dz ) + (nz/2) - 0.0f;
		break;

		case XZ_PLANE:

		//*a1 = ka1 * (iSlice - (ny/2) + 0.5f + (dx/2) - dd_sourcePosition[1]) + dd_sourcePosition[0] + (nx/2) - 0.0f;
		//*a2 = ka2 * (iSlice - (ny/2) + 0.5f + (dx/2) - dd_sourcePosition[1]) + dd_sourcePosition[0] + (nx/2) - 0.0f;

		*a1 = ka1 * (iSlice - (ny/2) +0.5f - dd_sourcePosition[1]) + dd_sourcePosition[0] + (nx/2) - 0.0f;
		*a2 = ka2 * (iSlice - (ny/2) +0.5f - dd_sourcePosition[1]) + dd_sourcePosition[0] + (nx/2) - 0.0f;

		//*z1 = ((kz1 * (iSlice - (ny/2) + 0.5f + (dz/2) - dd_sourcePosition[1]) + dd_sourcePosition[2]) / dz ) + (nz/2) - 0.0f;
		//*z2 = ((kz2 * (iSlice - (ny/2) + 0.5f + (dz/2) - dd_sourcePosition[1]) + dd_sourcePosition[2]) / dz ) + (nz/2) - 0.0f;
		*z1 = ((kz1 * (iSlice - (ny/2) +0.5f - dd_sourcePosition[1]) + dd_sourcePosition[2]) / dz ) + (nz/2) - 0.0f;
		*z2 = ((kz2 * (iSlice - (ny/2) +0.5f - dd_sourcePosition[1]) + dd_sourcePosition[2]) / dz ) + (nz/2) - 0.0f;

		break;
	}

	*A1 = floorf(*a1);
	*A2 = floorf(*a2);

	*Z1 = floorf(*z1);
	*Z2 = floorf(*z2);
}

// Calculate the weights for each of the voxels, read their intensity, multipy and sum
 __device__ float get_voxel_contributions(float A1, float A2, float a1, float a2, float Z1, float Z2, float z1, float z2, cudaTextureObject_t tex_volume, float nx, float ny, float nz, float slabIntersectionLength, int commonPlane, int iSlice){


	// Determine intersection type
	int intersectionType = 0;

	if ( (A1 == A2) && (Z1 == Z2)){
		intersectionType = 1;
	}
	else if ( (A1 != A2) && (Z1 == Z2)){
		intersectionType = 2;
	}
	else if ( (A1 == A2) && (Z1 != Z2)){
		intersectionType = 3;
	}
	else if ( (A1 != A2) && (Z1 != Z2)){
		intersectionType = 4;
	}


	float w = 0;
	float voxelContributions = 0;

switch (commonPlane){

// XZ Plane
	case XZ_PLANE:

	// Get weight and sum of voxel contributions
	switch (intersectionType){

		// only one voxel in the ith slice contributes to the detector cell
		case 1:
			if( (A1 >= 0) && (A1 <= ny - 1) && (Z1 >= 0) && (Z1 < nz)){
			w = 1;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f),(Z1 + 0.5f));
			}
			break;

		// two voxels in the ith slice in the horizontal direction contribute to the detector cell
		case 2:

			if( (A1 >= 0) && (A1 < ny) && (Z1 >= 0) && (Z1 < nz)){
			w = (MAX(A1, A2) - a1) / (a2 - a1);
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f), (Z1 + 0.5f));
			}

			if( (A2 >= 0) && (A2 < ny) && (Z1 >= 0) && (Z1 < nz)){
			w = (a2 - MAX(A1, A2)) / (a2 - a1);
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A2 + 0.5f, (iSlice + 0.5f), (Z1 + 0.5f));

			}
			break;

		// two voxels in the ith slice in the vertical direction contribute to the detector cell
		case 3:

			if( (A1 >= 0) && (A1 < (ny)) && (Z1 >= 0) && (Z1 < (nz))){
			w = (MAX(Z1, Z2) - z1) / (z2 - z1);
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f), (Z1 + 0.5f));
			}

			if( (A1 >= 0) && (A1 < (ny)) && (Z2 >= 0) && (Z2 < (nz))){
			w = (z2 - MAX(Z1,Z2)) / (z2 - z1);
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f), Z2 + 0.5f);
			}
			break;

		// four voxels in the ith slice contribute to the detector cell, 2 horizontal and 2 vertical
		case 4:
		
			if( (A1 >= 0) && (A1 < ny) && (Z1 >= 0) && (Z1 < nz) ){
			w = ( (MAX(A1,A2) - a1) / (a2 - a1) ) * ((MAX(Z1,Z2) - z1) / (z2 - z1));
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f), (Z1 + 0.5f));
			}

			if( (A2 >= 0) && (A2 < ny) && (Z1 >= 0) && (Z1 < nz)){
			w = ( (a2 - MAX(A1,A2)) / (a2 - a1) ) * ( (MAX(Z1,Z2) - z1) / (z2 - z1) );
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A2 + 0.5f, (iSlice + 0.5f), (Z1 + 0.5f));
			}


			if( (A1 >= 0) && (A1 < ny) && (Z2 >= 0) && (Z2 < nz)){
			w = ((MAX(A1,A2) - a1) / (a2 - a1) ) * ((z2 - MAX(Z1,Z2)) / (z2 - z1) );
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f), (Z2 + 0.5f));
			}


			if( (A2 >= 0) && (A2 < ny) && (Z2 >= 0) && (Z2 < nz)){
			w = ((a2 - MAX(A1,A2)) / (a2 - a1) ) * ((z2 - MAX(Z1,Z2)) / (z2 - z1));
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A2 + 0.5f, (iSlice + 0.5f), (Z2 + 0.5f));
			}
			break;
	}

	break;

// YZ Plane
	case YZ_PLANE:

	switch (intersectionType){

		// only one voxel in the ith slice contributes to the detector cell
		case 1:
			if( (A1 >= 0) && (A1 <= nx - 1) && (Z1 >= 0) && (Z1 < nz)){
			w = 1;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A1 + 0.5f, (Z1 + 0.5f));
			}
			break;

		// two voxels in the ith slice in the horizontal direction contribute to the detector cell (2 horizontal, 1 vertical)
		case 2:

			if( (A1 >= 0) && (A1 < nx) && (Z1 >= 0) && (Z1 < nz)){
			w = (MAX(A1, A2) - a1) / (a2 - a1);
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A1 + 0.5f, (Z1 + 0.5f));
			}

			if( (A2 >= 0) && (A2 < nx) && (Z1 >= 0) && (Z1 < nz)){
			w = (a2 - MAX(A1, A2)) / (a2 - a1);
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A2 + 0.5f, (Z1 + 0.5f));

			}
			break;

		// two voxels in the ith slice in the vertical direction contribute to the detector cell ( 1 horizontal, 2 vertical)
		case 3:

			if( (A1 >= 0) && (A1 < (nx)) && (Z1 >= 0) && (Z1 < (nz))){
			w = (MAX(Z1, Z2) - z1) / (z2 - z1);
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A1 + 0.5f, (Z1 + 0.5f));
			}

			if( (A1 >= 0) && (A1 < (nx)) && (Z2 >= 0) && (Z2 < (nz))){
			w = (z2 - MAX(Z1,Z2)) / (z2 - z1);
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), (A1 + 0.5f), Z2 + 0.5f);
			}
			break;

		// four voxels in the ith slice contribute to the detector cell, 2 horizontal and 2 vertical
		case 4:
		
			if( (A1 >= 0) && (A1 < nx) && (Z1 >= 0) && (Z1 < nz) ){

			w = ( (MAX(A1,A2) - a1) / (a2 - a1) ) * ((MAX(Z1,Z2) - z1) / (z2 - z1));
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A1 + 0.5f, (Z1 + 0.5f));
			}


			if( (A2 >= 0) && (A2 < nx) && (Z1 >= 0) && (Z1 < nz)){
			w = ( (a2 - MAX(A1,A2)) / (a2 - a1) ) * ( (MAX(Z1,Z2) - z1) / (z2 - z1) );
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A2 + 0.5f, (Z1 + 0.5f));
			}


			if( (A1 >= 0) && (A1 < nx) && (Z2 >= 0) && (Z2 < nz)){
			w = ((MAX(A1,A2) - a1) / (a2 - a1) ) * ((z2 - MAX(Z1,Z2)) / (z2 - z1) );
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A1 + 0.5f, (Z2 + 0.5f));
			}

			if( (A2 >= 0) && (A2 < nx) && (Z2 >= 0) && (Z2 < nz)){
			w = ((a2 - MAX(A1,A2)) / (a2 - a1) ) * ((z2 - MAX(Z1,Z2)) / (z2 - z1));
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A2 + 0.5f, (Z2 + 0.5f));
			}
			break;
	}


	break;
}

	return voxelContributions;
}

// version for SART that returns sum of projection weights
 __device__ float get_voxel_contributions(float A1, float A2, float a1, float a2, float Z1, float Z2, float z1, float z2, cudaTextureObject_t tex_volume, float nx, float ny, float nz, float slabIntersectionLength, int commonPlane, int iSlice, float* weightSum){
	// ...or how I learned to stop worrying and love the switch statement

	// Determine intersection type
	int intersectionType = 0;

	if ( (A1 == A2) && (Z1 == Z2)){
		intersectionType = 1;
	}
	else if ( (A1 != A2) && (Z1 == Z2)){
		intersectionType = 2;
	}
	else if ( (A1 == A2) && (Z1 != Z2)){
		intersectionType = 3;
	}
	else if ( (A1 != A2) && (Z1 != Z2)){
		intersectionType = 4;
	}


	float w = 0;
	float voxelContributions = 0;

switch (commonPlane){

// XZ Plane
	case XZ_PLANE:

	// Get weight and sum of voxel contributions
	switch (intersectionType){

		// only one voxel in the ith slice contributes to the detector cell
		case 1:
			if( (A1 >= 0) && (A1 <= ny - 1) && (Z1 >= 0) && (Z1 < nz)){
			w = 1;
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f),(Z1 + 0.5f));
			}
			break;

		// two voxels in the ith slice in the horizontal direction contribute to the detector cell
		case 2:

			if( (A1 >= 0) && (A1 < ny) && (Z1 >= 0) && (Z1 < nz)){
			w = (MAX(A1, A2) - a1) / (a2 - a1);
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f), (Z1 + 0.5f));
			}

			if( (A2 >= 0) && (A2 < ny) && (Z1 >= 0) && (Z1 < nz)){
			w = (a2 - MAX(A1, A2)) / (a2 - a1);
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A2 + 0.5f, (iSlice + 0.5f), (Z1 + 0.5f));

			}
			break;

		// two voxels in the ith slice in the vertical direction contribute to the detector cell
		case 3:

			if( (A1 >= 0) && (A1 < (ny)) && (Z1 >= 0) && (Z1 < (nz))){
			w = (MAX(Z1, Z2) - z1) / (z2 - z1);
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f), (Z1 + 0.5f));
			}

			if( (A1 >= 0) && (A1 < (ny)) && (Z2 >= 0) && (Z2 < (nz))){
			w = (z2 - MAX(Z1,Z2)) / (z2 - z1);
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f), Z2 + 0.5f);
			}
			break;

		// four voxels in the ith slice contribute to the detector cell, 2 horizontal and 2 vertical
		case 4:
		
			if( (A1 >= 0) && (A1 < ny) && (Z1 >= 0) && (Z1 < nz) ){
			w = ( (MAX(A1,A2) - a1) / (a2 - a1) ) * ((MAX(Z1,Z2) - z1) / (z2 - z1));
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f), (Z1 + 0.5f));
			}

			if( (A2 >= 0) && (A2 < ny) && (Z1 >= 0) && (Z1 < nz)){
			w = ( (a2 - MAX(A1,A2)) / (a2 - a1) ) * ( (MAX(Z1,Z2) - z1) / (z2 - z1) );
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A2 + 0.5f, (iSlice + 0.5f), (Z1 + 0.5f));
			}


			if( (A1 >= 0) && (A1 < ny) && (Z2 >= 0) && (Z2 < nz)){
			w = ((MAX(A1,A2) - a1) / (a2 - a1) ) * ((z2 - MAX(Z1,Z2)) / (z2 - z1) );
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A1 + 0.5f, (iSlice + 0.5f), (Z2 + 0.5f));
			}


			if( (A2 >= 0) && (A2 < ny) && (Z2 >= 0) && (Z2 < nz)){
			w = ((a2 - MAX(A1,A2)) / (a2 - a1) ) * ((z2 - MAX(Z1,Z2)) / (z2 - z1));
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, A2 + 0.5f, (iSlice + 0.5f), (Z2 + 0.5f));
			}
			break;
	}

	break;

// YZ Plane
	case YZ_PLANE:

	switch (intersectionType){

		// only one voxel in the ith slice contributes to the detector cell
		case 1:
			if( (A1 >= 0) && (A1 <= nx - 1) && (Z1 >= 0) && (Z1 < nz)){
			w = 1;
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A1 + 0.5f, (Z1 + 0.5f));
			}
			break;

		// two voxels in the ith slice in the horizontal direction contribute to the detector cell (2 horizontal, 1 vertical)
		case 2:

			if( (A1 >= 0) && (A1 < nx) && (Z1 >= 0) && (Z1 < nz)){
			w = (MAX(A1, A2) - a1) / (a2 - a1);
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A1 + 0.5f, (Z1 + 0.5f));
			}

			if( (A2 >= 0) && (A2 < nx) && (Z1 >= 0) && (Z1 < nz)){
			w = (a2 - MAX(A1, A2)) / (a2 - a1);
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A2 + 0.5f, (Z1 + 0.5f));

			}
			break;

		// two voxels in the ith slice in the vertical direction contribute to the detector cell ( 1 horizontal, 2 vertical)
		case 3:

			if( (A1 >= 0) && (A1 < (nx)) && (Z1 >= 0) && (Z1 < (nz))){
			w = (MAX(Z1, Z2) - z1) / (z2 - z1);
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A1 + 0.5f, (Z1 + 0.5f));
			}

			if( (A1 >= 0) && (A1 < (nx)) && (Z2 >= 0) && (Z2 < (nz))){
			w = (z2 - MAX(Z1,Z2)) / (z2 - z1);
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), (A1 + 0.5f), Z2 + 0.5f);
			}
			break;

		// four voxels in the ith slice contribute to the detector cell, 2 horizontal and 2 vertical
		case 4:
		
			if( (A1 >= 0) && (A1 < nx) && (Z1 >= 0) && (Z1 < nz) ){

			w = ( (MAX(A1,A2) - a1) / (a2 - a1) ) * ((MAX(Z1,Z2) - z1) / (z2 - z1));
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A1 + 0.5f, (Z1 + 0.5f));
			}


			if( (A2 >= 0) && (A2 < nx) && (Z1 >= 0) && (Z1 < nz)){
			w = ( (a2 - MAX(A1,A2)) / (a2 - a1) ) * ( (MAX(Z1,Z2) - z1) / (z2 - z1) );
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A2 + 0.5f, (Z1 + 0.5f));
			}


			if( (A1 >= 0) && (A1 < nx) && (Z2 >= 0) && (Z2 < nz)){
			w = ((MAX(A1,A2) - a1) / (a2 - a1) ) * ((z2 - MAX(Z1,Z2)) / (z2 - z1) );
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A1 + 0.5f, (Z2 + 0.5f));
			}

			if( (A2 >= 0) && (A2 < nx) && (Z2 >= 0) && (Z2 < nz)){
			w = ((a2 - MAX(A1,A2)) / (a2 - a1) ) * ((z2 - MAX(Z1,Z2)) / (z2 - z1));
			*weightSum += w;
			voxelContributions += w * slabIntersectionLength * tex3D<float>(tex_volume, (iSlice + 0.5f), A2 + 0.5f, (Z2 + 0.5f));
			}
			break;
	}


	break;
}

	return voxelContributions;
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

__device__ float forward_project(cudaTextureObject_t tex_volume, float* hd_sourcePhi, float* hd_sourceZ, int u, int v, int t, float SI, float DI, float nu, float nv, float du, float dv, float nx, float ny, float nz, float* weightSum){

	// Arrays to hold coordinates
	float dd_sourcePosition[3];
	float dd_detectorPosition[3];
	float dd_centralDetectorPosition[3];
	float dd_u1[3];
	float dd_v1[3];

	// sin and cos of projection ange (with respect to x-axis)
	float sourcePhi;
	float cosPhi, sinPhi;
	float* dd_cos = &cosPhi;
	float* dd_sin = &sinPhi;

	// length of intersection between line from detector element center to source and each XZ or YZ slabc:w
	float slabIntersectionLength;

	// Detector boundary postions (first and last in u,v directions)
	float dd_uF[3];
	float dd_uL[3];
	float dd_vF[3];
	float dd_vL[3];

	// slopes (depends on which common plane is used)
	float ka1, ka2;

	float kz1, kz2;
	// (x or y) indices of first and last voxel that influence this detector
	float A1, A2;
	// (x or y) projected coordinates of first and last voxel
	float a1, a2;

	// z indices of first and last voxel that influence this detector
	float Z1, Z2;
	// z projected coordinates of first and last voxel
	float z1, z2;

	// Source z position for helical
	float sourceZ = 0;
	if (hd_sourceZ == NULL){
		sourceZ = 0;
	}
	else{
		sourceZ = hd_sourceZ[t];
	}

	// See preprocessor macro at top
	int commonPlane = -1;

	// Value for this detector for this time point
	float voxelContributions = 0;

	// Sum of weights for SART
	int normalize = -1;

	if(weightSum == NULL){
		normalize = 0;
	}
	else{
		normalize = 1;
	}

	// Get projection angle 
	sourcePhi = hd_sourcePhi[t];

	// cos and sin of projection angle
	__sincosf(sourcePhi, dd_sin, dd_cos); 

	// Get geometry for this view
	get_source_position(dd_sourcePosition, cosPhi, sinPhi, SI, sourceZ);
	get_central_detector_array_position(dd_centralDetectorPosition, dd_sourcePosition, cosPhi, sinPhi, DI, sourceZ);
	get_detector_basis_vectors_flat(dd_sourcePosition, dd_u1, dd_v1);
	get_detector_position_flat(dd_detectorPosition, dd_centralDetectorPosition, dd_u1, dd_v1, u, v, nu, nv, du, dv);
	get_detector_boundary_positions_flat(dd_uF, dd_uL, dd_vF, dd_vL, dd_detectorPosition, dd_v1, du,  dv, cosPhi, sinPhi);
	 
	// Determine which plane to project detector and voxel boundaries onto
	commonPlane = get_common_plane(dd_sourcePosition, dd_detectorPosition);

	// Find slopes of lines from detector boundaries to source
	get_slopes(&ka1, &ka2, &kz1, &kz2, dd_uF, dd_uL, dd_vF, dd_vL, dd_sourcePosition, commonPlane);

	switch (commonPlane){

		case XZ_PLANE :

			slabIntersectionLength = get_slab_intersection_length(dd_sourcePosition, dd_detectorPosition, commonPlane, u, du, nu);
			// Loop over y slices
			for(int iSlice = 0; iSlice < (int)ny; iSlice++){

			get_intersecting_voxels(&A1, &A2, &a1, &a2, &Z1, &Z2, &z1, &z2, ka1, ka2, kz1, kz2, dd_sourcePosition, nx, ny, nz, du, dv, commonPlane, iSlice);

			if(normalize){
			voxelContributions += get_voxel_contributions(A1, A2, a1, a2, Z1, Z2, z1, z2, tex_volume, nx, nx, nz, slabIntersectionLength, commonPlane, iSlice, weightSum);
			}
			else{
			voxelContributions += get_voxel_contributions(A1, A2, a1, a2, Z1, Z2, z1, z2, tex_volume, nx, nx, nz, slabIntersectionLength, commonPlane, iSlice);
			}


			}

			break;

		case YZ_PLANE:
			
			slabIntersectionLength = get_slab_intersection_length(dd_sourcePosition, dd_detectorPosition, commonPlane, u, du, nu);
			// Loop over x slices
			for(int iSlice = 0; iSlice < (int)nx; iSlice++){

			get_intersecting_voxels(&A1, &A2, &a1, &a2, &Z1, &Z2, &z1, &z2, ka1, ka2, kz1, kz2, dd_sourcePosition, nx, ny, nz, du, dv, commonPlane, iSlice);

			if(normalize){
			voxelContributions += get_voxel_contributions(A1, A2, a1, a2, Z1, Z2, z1, z2, tex_volume, nx, nx, nz, slabIntersectionLength, commonPlane, iSlice, weightSum);
			}
			else{
			voxelContributions += get_voxel_contributions(A1, A2, a1, a2, Z1, Z2, z1, z2, tex_volume, nx, nx, nz, slabIntersectionLength, commonPlane, iSlice);
			}


		}
			break;
	}


return voxelContributions;
// end forward projection
}

// forward projection kernel
__global__ void forward_projection_kernel(float* hd_projections, cudaTextureObject_t tex_volume, float* hd_sourcePhi, float* hd_sourceZ){
		
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;
	int t = (blockIdx.z * blockDim.z) + threadIdx.z;

	// Get geometry constants from constant memory
	float SI = hdc_geometry[0];
	float DI = hdc_geometry[1];
	float nu = hdc_geometry[2];
	float nv = hdc_geometry[3];
	float du = hdc_geometry[4];
	float dv = hdc_geometry[5];
	float nx = hdc_geometry[6];
	float ny = hdc_geometry[7];
	float nz = hdc_geometry[8];
	float nt = hdc_geometry[9];

	// Bounds check
	if ( u > ((int)nu - 1) || v > ((int)nv - 1) || t > ((int)nt - 1)){
		return;
	}

	// Get linear index into projection output
	unsigned int ind_global = u + (v * (int) nu) + (t * (int)nu * (int)nv);

	// Forward project
	float voxelContributions = 0;
       	voxelContributions = forward_project(tex_volume, hd_sourcePhi, hd_sourceZ, u, v, t, SI, DI, nu, nv, du, dv, nx, ny, nz, NULL);

// Write result
hd_projections[ind_global] = voxelContributions;
}


// forward launcher
void dist_forward_project(float* hh_projections, float* hh_volume, float* hh_sourcePhi, float* hh_sourceZ, float* hh_SI, float* hh_DI, float* hh_nu, float* hh_nv, float* hh_du, float* hh_dv, size_t nx, size_t ny, size_t nz, size_t nt, size_t nVoxels, float* hh_geometry){

float nu = *hh_nu;
float nv = *hh_nv;

// Calculate memory allocation sizes
size_t nBytesTimepointVector = nt * sizeof(float);
size_t nBytesProjections = nu * nv * nt * sizeof(float);
size_t nBytesGeometry = N_GEOMETRY_CONSTANTS * sizeof(float);

// Allocate CUDA array for image volume 
cudaArray* hda_volume;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
struct cudaExtent extent = make_cudaExtent(nx, ny, nz);
cudaCheck(cudaMalloc3DArray(&hda_volume, &channelDesc, extent));
cudaPitchedPtr hhp_volume = make_cudaPitchedPtr((void*) hh_volume, nx * sizeof(float), nx, ny);

// Copy volume to device
cudaMemcpy3DParms copyParams = {0};
copyParams.srcPtr = hhp_volume;
copyParams.dstArray = hda_volume;
copyParams.extent = extent;
copyParams.kind = cudaMemcpyHostToDevice;
cudaCheck(cudaMemcpy3D(&copyParams));

// Create texture objects
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = hda_volume;

cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeBorder;
texDesc.addressMode[1] = cudaAddressModeBorder;
texDesc.addressMode[2] = cudaAddressModeBorder;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;

cudaTextureObject_t tex_volume = 0;
cudaCreateTextureObject(&tex_volume, &resDesc, &texDesc, NULL);

// Allocate global memory for remaining arrays and copy to device
float* hd_sourcePhi;
cudaCheck(cudaMalloc((void**)&hd_sourcePhi, nBytesTimepointVector));
cudaCheck(cudaMemcpy(hd_sourcePhi, hh_sourcePhi, nBytesTimepointVector, cudaMemcpyHostToDevice));

// Move zPositions over to GPU if we're doing a helical scan
float* hd_sourceZ;
if(hh_sourceZ == NULL){
	hd_sourceZ = NULL;
}
else{
cudaCheck(cudaMalloc((void**)&hd_sourceZ, nBytesTimepointVector));
cudaCheck(cudaMemcpy(hd_sourceZ, hh_sourceZ, nBytesTimepointVector, cudaMemcpyHostToDevice));
}

// Copy geoemtric distances to constant memory
hh_geometry[0] = *hh_SI;
hh_geometry[1] = *hh_DI;
hh_geometry[2] = *hh_nu;
hh_geometry[3] = *hh_nv;
hh_geometry[4] = *hh_du;
hh_geometry[5] = *hh_dv;
hh_geometry[6] = (float) nx;
hh_geometry[7] = (float) ny;
hh_geometry[8] = (float) nz;
hh_geometry[9] = (float) nt;

cudaCheck(cudaMemcpyToSymbol(hdc_geometry, hh_geometry, nBytesGeometry));

// Allocate device memory for projections
float* hd_projections;
cudaCheck(cudaMalloc((void**)&hd_projections, nBytesProjections));

// Allocate device memory for weight sums 

// Determine grid size
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT,BLOCKDEPTH);
const dim3 gridSize(nu/BLOCKWIDTH + 1, nv/BLOCKWIDTH + 1, nt/BLOCKDEPTH + 1);

// Kernel launch
forward_projection_kernel<<<gridSize,blockSize>>>(hd_projections, tex_volume, hd_sourcePhi, hd_sourceZ);

cudaCheck(cudaDeviceSynchronize());
//cudaCheck(cudaPeekAtLastError());

// Copy back to host
cudaCheck(cudaMemcpy(hh_projections, hd_projections, nBytesProjections, cudaMemcpyDeviceToHost));

// Free allocated memory
cudaDestroyTextureObject(tex_volume);
cudaFreeArray(hda_volume);
cudaFree(hd_projections);
cudaFree(hd_sourcePhi);
cudaFree(hd_sourceZ);

//Reset device for profiling
cudaDeviceReset();
return;
}

__device__ float back_project(cudaTextureObject_t tex_projections, float* hd_sourcePhi, float* hd_sourceZ, int x, int y, int z, float SI, float DI, float nu, float nv, float du, float dv, float nx, float ny, float nz, int nt, float* weightSum){

	// world coordinates for this voxel
	float dd_voxel[3];
	dd_voxel[0] = (float)x - (nx/2);
	dd_voxel[1] = (float)y - (ny/2);
	dd_voxel[2] = (float)z - (nz/2);

	dd_voxel[0] = dd_voxel[0] + 0.5f;
	dd_voxel[1] = dd_voxel[1] + 0.5f;
	dd_voxel[2] = dd_voxel[2] + 0.5f;

	// voxel boundary coordinates
	float a1[3];
	float a2[3];
	float z1[3];
	float z2[3];

	//intermediate variables
	float voxelSum = 0;
	float uMin, uMax;
	float vMin, vMax;
	float dd_uv[2];

	float uMinInd, uMaxInd;
	int vMinInd, vMaxInd;

	float uBound1, uBound2;
	float vBound1, vBound2;

	float cosPhi, sinPhi;
	float* dd_cos = &cosPhi;
	float* dd_sin = &sinPhi;

	float wu, wv, w;

	float dd_centralDetectorPosition_rotated[3];
	float dd_sourcePosition_rotated[3];

	float dd_intersection[3];

	float u1_rotated[3];
	float v1_rotated[3];
	int dd_uvInd[2];

	// Helical scan?
	int helical = -1;
	float sourceZ = 0;
	float dd_helical_detector_vector[3];

	if(hd_sourceZ == NULL){
		helical = 0;
		sourceZ = 0;
	}
	else{
		helical = 1;
	}

	// Normalize? 
	int normalize = -1;
	if(weightSum == NULL){
		normalize = 0;
	}
	else{
		normalize = 1;
	}

	float tmp;

	//Loop counters
	int iV, iU;

	// Rotate to phi = 0
	sinPhi = 0.0f;
	cosPhi = 1.0f;

	dd_centralDetectorPosition_rotated[0] = DI * - 1;
	dd_centralDetectorPosition_rotated[1] = 0;
	dd_centralDetectorPosition_rotated[2] = sourceZ;

	dd_sourcePosition_rotated[0] = SI;
	dd_sourcePosition_rotated[1] = 0;
	dd_sourcePosition_rotated[2] = sourceZ;

	u1_rotated[0] = 0.0f;
	u1_rotated[1] = 0.0f;
	u1_rotated[2] = 1.0f;

	v1_rotated[0] = 0.0f;
	v1_rotated[1] = 1.0f;
	v1_rotated[2] = 0.0f;

	dd_helical_detector_vector[0] = dd_centralDetectorPosition_rotated[0];
	dd_helical_detector_vector[1] = dd_centralDetectorPosition_rotated[1];
	dd_helical_detector_vector[2] = 0;

        for(int iAngle = 0; iAngle < nt; iAngle++){

	// cos and sin of projection angle
	__sincosf(hd_sourcePhi[iAngle], dd_sin, dd_cos); 

	// get rotated coordinates of voxel edges:
	//left
	a1[0] = dd_voxel[0] * cosPhi + dd_voxel[1] * sinPhi;
	a1[1] = dd_voxel[0] * -1 * sinPhi + dd_voxel[1] * cosPhi;
	a1[2] = dd_voxel[2];

	a1[1] = a1[1] - 0.5f;
	a1[0] = a1[0] - 0.5f;

	// right
	a2[0] = dd_voxel[0] * cosPhi + dd_voxel[1] * sinPhi;
	a2[1] = dd_voxel[0] * -1 * sinPhi + dd_voxel[1] * cosPhi;
	a2[2] = dd_voxel[2];

	a2[1] = a2[1] + 0.5f;
	a2[0] = a2[0] + 0.5f;

	//lower
	z1[0] = (dd_voxel[0] * cosPhi) + (dd_voxel[1] * sinPhi);
	z1[1] = -1 * (dd_voxel[0] * sinPhi) + (dd_voxel[1] * cosPhi);
	z1[2] = dd_voxel[2] - (dz/2);

	//upper
	z2[0] = (dd_voxel[0] * cosPhi) + (dd_voxel[1] * sinPhi);
	z2[1] = -1 * (dd_voxel[0] * sinPhi) + (dd_voxel[1] * cosPhi);
	z2[2] = dd_voxel[2] + (dz/2);

	// helical scan?
	if(helical == 1){
	sourceZ = hd_sourceZ[iAngle];
	dd_sourcePosition_rotated[2] = sourceZ;
	dd_centralDetectorPosition_rotated[2] = sourceZ;
	}

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

		if (normalize){
		*weightSum += w;
		}

		voxelSum += w * tex3D<float>(tex_projections,(iU + 0.5f), (iV + 0.5f), (iAngle + 0.5f));

// end loop over detectors
	}
	}

//end loop over angles
	}
return voxelSum;
}


// magnification for SAME
__device__ float back_project_mag(cudaTextureObject_t tex_projections, float* hd_sourcePhi, float* hd_sourceZ, int x, int y, int z, float SI, float DI, float nu, float nv, float du, float dv, float nx, float ny, float nz, int nt, float* weightSum){

	// world coordinates for this voxel
	float dd_voxel[3];
	dd_voxel[0] = (float)x - (nx/2);
	dd_voxel[1] = (float)y - (ny/2);
	dd_voxel[2] = (float)z - (nz/2);

	dd_voxel[0] = dd_voxel[0] + 0.5f;
	dd_voxel[1] = dd_voxel[1] + 0.5f;
	dd_voxel[2] = dd_voxel[2] + 0.5f;

	// voxel boundary coordinates
	float a1[3];
	float a2[3];
	float z1[3];
	float z2[3];

	//intermediate variables
	float voxelSum = 0;
	float uMin, uMax;
	float vMin, vMax;
	float dd_uv[2];

	float uMinInd, uMaxInd;
	int vMinInd, vMaxInd;

	float uBound1, uBound2;
	float vBound1, vBound2;

	float cosPhi, sinPhi;
	float* dd_cos = &cosPhi;
	float* dd_sin = &sinPhi;

	float wu, wv, w;

	float dd_centralDetectorPosition_rotated[3];
	float dd_sourcePosition_rotated[3];

	float dd_intersection[3];

	float u1_rotated[3];
	float v1_rotated[3];
	int dd_uvInd[2];

	float dd_detectorPosition[3];
	// Helical scan?
	int helical = -1;
	float sourceZ = 0;
	float dd_helical_detector_vector[3];

	if(hd_sourceZ == NULL){
		helical = 0;
		sourceZ = 0;
	}
	else{
		helical = 1;
	}

	// Normalize? 
	int normalize = -1;
	if(weightSum == NULL){
		normalize = 0;
	}
	else{
		normalize = 1;
	}

	float tmp;

	//Loop counters
	int iV, iU;

	float SID,SOD,m;

	// Rotate to phi = 0
	sinPhi = 0.0f;
	cosPhi = 1.0f;

	dd_centralDetectorPosition_rotated[0] = DI * - 1;
	dd_centralDetectorPosition_rotated[1] = 0;
	dd_centralDetectorPosition_rotated[2] = sourceZ;

	dd_sourcePosition_rotated[0] = SI;
	dd_sourcePosition_rotated[1] = 0;
	dd_sourcePosition_rotated[2] = sourceZ;

	u1_rotated[0] = 0.0f;
	u1_rotated[1] = 0.0f;
	u1_rotated[2] = 1.0f;

	v1_rotated[0] = 0.0f;
	v1_rotated[1] = 1.0f;
	v1_rotated[2] = 0.0f;

	dd_helical_detector_vector[0] = dd_centralDetectorPosition_rotated[0];
	dd_helical_detector_vector[1] = dd_centralDetectorPosition_rotated[1];
	dd_helical_detector_vector[2] = 0;

        for(int iAngle = 0; iAngle < nt; iAngle++){

	// cos and sin of projection angle
	__sincosf(hd_sourcePhi[iAngle], dd_sin, dd_cos); 

	// get rotated coordinates of voxel edges:
	//left
	a1[0] = dd_voxel[0] * cosPhi + dd_voxel[1] * sinPhi;
	a1[1] = dd_voxel[0] * -1 * sinPhi + dd_voxel[1] * cosPhi;
	a1[2] = dd_voxel[2];

	a1[1] = a1[1] - 0.5f;
	a1[0] = a1[0] - 0.5f;

	// right
	a2[0] = dd_voxel[0] * cosPhi + dd_voxel[1] * sinPhi;
	a2[1] = dd_voxel[0] * -1 * sinPhi + dd_voxel[1] * cosPhi;
	a2[2] = dd_voxel[2];

	a2[1] = a2[1] + 0.5f;
	a2[0] = a2[0] + 0.5f;

	//lower
	z1[0] = (dd_voxel[0] * cosPhi) + (dd_voxel[1] * sinPhi);
	z1[1] = -1 * (dd_voxel[0] * sinPhi) + (dd_voxel[1] * cosPhi);
	z1[2] = dd_voxel[2] - (dz/2);

	//upper
	z2[0] = (dd_voxel[0] * cosPhi) + (dd_voxel[1] * sinPhi);
	z2[1] = -1 * (dd_voxel[0] * sinPhi) + (dd_voxel[1] * cosPhi);
	z2[2] = dd_voxel[2] + (dz/2);

	// helical scan?
	if(helical == 1){
	sourceZ = hd_sourceZ[iAngle];
	dd_sourcePosition_rotated[2] = sourceZ;
	dd_centralDetectorPosition_rotated[2] = sourceZ;
	}

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

		if (normalize){
		*weightSum += w;
		}
	
		
	get_detector_position_flat(dd_detectorPosition, dd_centralDetectorPosition_rotated, u1_rotated, v1_rotated, (unsigned int) iU, (unsigned int) iV, nu, nv, du, dv);

	// calculate magnification
	SID = (dd_sourcePosition_rotated[0] - dd_detectorPosition[0]) * (dd_sourcePosition_rotated[0] - dd_detectorPosition[0]) + (dd_sourcePosition_rotated[1] - dd_detectorPosition[1]) * (dd_sourcePosition_rotated[1] - dd_detectorPosition[1]) + (dd_sourcePosition_rotated[2] - dd_detectorPosition[2]) * (dd_sourcePosition_rotated[2] - dd_detectorPosition[2]);
	SID = sqrtf(SID);

	SOD = (dd_sourcePosition_rotated[0] - dd_voxel[0]) * (dd_sourcePosition_rotated[0] - dd_voxel[0]) + (dd_sourcePosition_rotated[1] - dd_voxel[1]) * (dd_sourcePosition_rotated[1] - dd_voxel[1]) + (dd_sourcePosition_rotated[2] - dd_voxel[2]) * (dd_sourcePosition_rotated[2] - dd_voxel[2]);
	SOD = sqrtf(SOD);

	m = SID/SOD;
	m = 1/m;

	voxelSum += w * m * tex3D<float>(tex_projections,(iU + 0.5f), (iV + 0.5f), (iAngle + 0.5f));

// end loop over detectors
	}
	}

//end loop over angles
	}
return voxelSum;
}


// Backprojection kernel
__global__ void back_projection_kernel(float* hd_volume, cudaTextureObject_t tex_projections, float* hd_sourcePhi, float* hd_sourceZ){

	int x = (int) (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (int) (blockIdx.y * blockDim.y) + threadIdx.y;
	int z = (int) (blockIdx.z * blockDim.z) + threadIdx.z;

	// Get geometry constants from constant memory
	float SI = hdc_geometry[0];
	float DI = hdc_geometry[1];
	float nu = hdc_geometry[2];
	float nv = hdc_geometry[3];
	float du = hdc_geometry[4];
	float dv = hdc_geometry[5];
	float nx = hdc_geometry[6];
	float ny = hdc_geometry[7];
	float nz = hdc_geometry[8];
	float nt = hdc_geometry[9];

	// Bounds check
	if ( x > ((int)nx - 1) || y > ((int)ny - 1) || z > ((int)nz - 1)){
		return;
	}

	// Get linear index into projection output
	size_t ind_global = x + (y * (int)nx) + (z * (int)nx * (int)ny);

	float voxelSum = 0;
	voxelSum = back_project(tex_projections, hd_sourcePhi, hd_sourceZ, x, y, z, SI, DI, nu, nv, du, dv, nx, ny, nz, nt, NULL);

	hd_volume[ind_global] = voxelSum;
}

__global__ void back_projection_kernel_mag(float* hd_volume, cudaTextureObject_t tex_projections, float* hd_sourcePhi, float* hd_sourceZ){

	int x = (int) (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (int) (blockIdx.y * blockDim.y) + threadIdx.y;
	int z = (int) (blockIdx.z * blockDim.z) + threadIdx.z;

	// Get geometry constants from constant memory
	float SI = hdc_geometry[0];
	float DI = hdc_geometry[1];
	float nu = hdc_geometry[2];
	float nv = hdc_geometry[3];
	float du = hdc_geometry[4];
	float dv = hdc_geometry[5];
	float nx = hdc_geometry[6];
	float ny = hdc_geometry[7];
	float nz = hdc_geometry[8];
	float nt = hdc_geometry[9];

	// Bounds check
	if ( x > ((int)nx - 1) || y > ((int)ny - 1) || z > ((int)nz - 1)){
		return;
	}

	// Get linear index into projection output
	size_t ind_global = x + (y * (int)nx) + (z * (int)nx * (int)ny);

	float voxelSum = 0;
	voxelSum = back_project_mag(tex_projections, hd_sourcePhi, hd_sourceZ, x, y, z, SI, DI, nu, nv, du, dv, nx, ny, nz, nt, NULL);

	hd_volume[ind_global] = voxelSum;
}


//back projection launcher

void dist_back_project(float* hh_volume, float* hh_projections, float* hh_sourcePhi, float* hh_sourceZ, float* hh_SI, float* hh_DI, float* hh_du, float* hh_dv, size_t nu, size_t nv, size_t nx, size_t ny, size_t nz, size_t nt, size_t nBytesTimepointVector, float* hh_geometry, size_t nBytesGeometry, size_t nBytesVolume){


// Allocate CUDA array for projection data 
cudaArray* hda_projections;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
struct cudaExtent extent = make_cudaExtent(nu, nv, nt);
//struct cudaExtent extent = make_cudaExtent(nv, nu, nt);

cudaCheck(cudaMalloc3DArray(&hda_projections, &channelDesc, extent));
cudaPitchedPtr hhp_projections = make_cudaPitchedPtr((void*) hh_projections, nu * sizeof(float), nu, nv);
//cudaPitchedPtr hhp_projections = make_cudaPitchedPtr((void*) hh_projections, nv * sizeof(float), nv, nu);

// Copy volume to device
cudaMemcpy3DParms copyParams = {0};
copyParams.srcPtr = hhp_projections;
copyParams.dstArray = hda_projections;
copyParams.extent = extent;
copyParams.kind = cudaMemcpyHostToDevice;
cudaCheck(cudaMemcpy3D(&copyParams));

// Create texture object
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = hda_projections;

cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeBorder;
texDesc.addressMode[1] = cudaAddressModeBorder;
texDesc.addressMode[2] = cudaAddressModeBorder;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;

cudaTextureObject_t tex_projections = 0;
cudaCreateTextureObject(&tex_projections, &resDesc, &texDesc, NULL);

// Allocate global memory for remaining arrays and copy to device
float* hd_sourcePhi;
cudaCheck(cudaMalloc((void**)&hd_sourcePhi, nBytesTimepointVector));
cudaCheck(cudaMemcpy(hd_sourcePhi, hh_sourcePhi, nBytesTimepointVector, cudaMemcpyHostToDevice));

float* hd_sourceZ;

if(hh_sourceZ == NULL){
	hd_sourceZ = NULL;
}
else{
cudaCheck(cudaMalloc((void**)&hd_sourceZ, nBytesTimepointVector));
cudaCheck(cudaMemcpy(hd_sourceZ, hh_sourceZ, nBytesTimepointVector, cudaMemcpyHostToDevice));
}

// Copy geoemtric distances to constant memory
hh_geometry[0] = *hh_SI;
hh_geometry[1] = *hh_DI;
hh_geometry[2] = (float) nu;
hh_geometry[3] = (float) nv; 
hh_geometry[4] = *hh_du;
hh_geometry[5] = *hh_dv;
hh_geometry[6] = (float) nx;
hh_geometry[7] = (float) ny;
hh_geometry[8] = (float) nz;
hh_geometry[9] = (float) nt;

cudaCheck(cudaMemcpyToSymbol(hdc_geometry, hh_geometry, nBytesGeometry));

// Allocate device memory for output image volume
float* hd_volume;
cudaCheck(cudaMalloc((void**)&hd_volume, nBytesVolume));

// Determine grid size
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT,BLOCKDEPTH);
const dim3 gridSize( (nx/BLOCKWIDTH) + 1, (ny/BLOCKWIDTH) + 1, (nz/BLOCKDEPTH) + 1);

// Kernel launch
back_projection_kernel<<<gridSize,blockSize>>>(hd_volume, tex_projections, hd_sourcePhi, hd_sourceZ);
cudaCheck(cudaDeviceSynchronize());
//cudaCheck(cudaPeekAtLastError());

// Copy back to host
cudaCheck(cudaMemcpy(hh_volume, hd_volume, nBytesVolume, cudaMemcpyDeviceToHost));

//mexPrintf("%0.2f\n",hh_volume[16777217]);

// Free allocated memory
cudaDestroyTextureObject(tex_projections);
cudaFreeArray(hda_projections);
cudaFree(hd_volume);
cudaFree(hd_sourcePhi);
cudaFree(hd_sourceZ);

//Reset device for profiling
cudaDeviceReset();
return;
}

void dist_back_project_mag(float* hh_volume, float* hh_projections, float* hh_sourcePhi, float* hh_sourceZ, float* hh_SI, float* hh_DI, float* hh_du, float* hh_dv, size_t nu, size_t nv, size_t nx, size_t ny, size_t nz, size_t nt, size_t nBytesTimepointVector, float* hh_geometry, size_t nBytesGeometry, size_t nBytesVolume){


// Allocate CUDA array for projection data 
cudaArray* hda_projections;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
struct cudaExtent extent = make_cudaExtent(nu, nv, nt);
//struct cudaExtent extent = make_cudaExtent(nv, nu, nt);

cudaCheck(cudaMalloc3DArray(&hda_projections, &channelDesc, extent));
cudaPitchedPtr hhp_projections = make_cudaPitchedPtr((void*) hh_projections, nu * sizeof(float), nu, nv);
//cudaPitchedPtr hhp_projections = make_cudaPitchedPtr((void*) hh_projections, nv * sizeof(float), nv, nu);

// Copy volume to device
cudaMemcpy3DParms copyParams = {0};
copyParams.srcPtr = hhp_projections;
copyParams.dstArray = hda_projections;
copyParams.extent = extent;
copyParams.kind = cudaMemcpyHostToDevice;
cudaCheck(cudaMemcpy3D(&copyParams));

// Create texture object
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = hda_projections;

cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeBorder;
texDesc.addressMode[1] = cudaAddressModeBorder;
texDesc.addressMode[2] = cudaAddressModeBorder;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;

cudaTextureObject_t tex_projections = 0;
cudaCreateTextureObject(&tex_projections, &resDesc, &texDesc, NULL);

// Allocate global memory for remaining arrays and copy to device
float* hd_sourcePhi;
cudaCheck(cudaMalloc((void**)&hd_sourcePhi, nBytesTimepointVector));
cudaCheck(cudaMemcpy(hd_sourcePhi, hh_sourcePhi, nBytesTimepointVector, cudaMemcpyHostToDevice));

float* hd_sourceZ;

if(hh_sourceZ == NULL){
	hd_sourceZ = NULL;
}
else{
cudaCheck(cudaMalloc((void**)&hd_sourceZ, nBytesTimepointVector));
cudaCheck(cudaMemcpy(hd_sourceZ, hh_sourceZ, nBytesTimepointVector, cudaMemcpyHostToDevice));
}

// Copy geoemtric distances to constant memory
hh_geometry[0] = *hh_SI;
hh_geometry[1] = *hh_DI;
hh_geometry[2] = (float) nu;
hh_geometry[3] = (float) nv; 
hh_geometry[4] = *hh_du;
hh_geometry[5] = *hh_dv;
hh_geometry[6] = (float) nx;
hh_geometry[7] = (float) ny;
hh_geometry[8] = (float) nz;
hh_geometry[9] = (float) nt;

cudaCheck(cudaMemcpyToSymbol(hdc_geometry, hh_geometry, nBytesGeometry));

// Allocate device memory for output image volume
float* hd_volume;
cudaCheck(cudaMalloc((void**)&hd_volume, nBytesVolume));

// Determine grid size
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT,BLOCKDEPTH);
const dim3 gridSize( (nx/BLOCKWIDTH) + 1, (ny/BLOCKWIDTH) + 1, (nz/BLOCKDEPTH) + 1);

// Kernel launch
back_projection_kernel_mag<<<gridSize,blockSize>>>(hd_volume, tex_projections, hd_sourcePhi, hd_sourceZ);
cudaCheck(cudaDeviceSynchronize());
//cudaCheck(cudaPeekAtLastError());

// Copy back to host
cudaCheck(cudaMemcpy(hh_volume, hd_volume, nBytesVolume, cudaMemcpyDeviceToHost));

//mexPrintf("%0.2f\n",hh_volume[16777217]);

// Free allocated memory
cudaDestroyTextureObject(tex_projections);
cudaFreeArray(hda_projections);
cudaFree(hd_volume);
cudaFree(hd_sourcePhi);
cudaFree(hd_sourceZ);

//Reset device for profiling
cudaDeviceReset();
return;
}
// *** SART ***
__global__ void corrective_forward_kernel(cudaSurfaceObject_t surf_correctiveProjections, cudaTextureObject_t tex_projections, cudaTextureObject_t tex_volume, float* hd_sourcePhi, float* hd_sourceZ, float lamda, int ntSubset, int subsetOffset){

	// Get geometry constants from constant memory
	float SI = hdc_geometry[0];
	float DI = hdc_geometry[1];
	float nu = hdc_geometry[2];
	float nv = hdc_geometry[3];
	float du = hdc_geometry[4];
	float dv = hdc_geometry[5];
	float nx = hdc_geometry[6];
	float ny = hdc_geometry[7];
	float nz = hdc_geometry[8];
//	float nt = hdc_geometry[9];

	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	int v = (blockIdx.y * blockDim.y) + threadIdx.y;
	int t = (blockIdx.z * blockDim.z) + threadIdx.z;

	// Bounds check
	if ( u > ((int)nu - 1) || v > ((int)nv - 1) || t > (ntSubset - 1)){
		return;
	}

	// Get linear index into projection output
	//unsigned int ind_global = u + (v * (int) nu) + (t * (int)nu * (int)nv);

	float correctiveProjection = 0;
	float weightSum = 0;
	
	// Forward project through current volume
	correctiveProjection = forward_project(tex_volume, hd_sourcePhi, hd_sourceZ, u, v, t, SI, DI, nu, nv, du, dv, nx, ny, nz, &weightSum);

	// Read measured projection
	float actualProjectionValue = 0;
	actualProjectionValue = tex3D<float>(tex_projections,(u + 0.5f), (v + 0.5f), ((subsetOffset + t) + 0.5f));

	// Calculate correction
	if(weightSum == 0){
		correctiveProjection = 0;
	}
	else{
		correctiveProjection = (actualProjectionValue - correctiveProjection) / weightSum;
	}

	// Write result
	surf3Dwrite(correctiveProjection, surf_correctiveProjections, u * sizeof(float), v, t);
}


__global__ void corrective_back_kernel(cudaSurfaceObject_t surf_correctiveVolume, cudaTextureObject_t tex_volume, cudaTextureObject_t tex_correctiveProjections, float* hd_sourcePhi, float* hd_sourceZ, float lamda, int ntSubset){

	// Get geometry constants from constant memory
	float SI = hdc_geometry[0];
	float DI = hdc_geometry[1];
	float nu = hdc_geometry[2];
	float nv = hdc_geometry[3];
	float du = hdc_geometry[4];
	float dv = hdc_geometry[5];
	float nx = hdc_geometry[6];
	float ny = hdc_geometry[7];
	float nz = hdc_geometry[8];
	//float nt = hdc_geometry[9];

	// Calculate thread index
	int x = (int) (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (int) (blockIdx.y * blockDim.y) + threadIdx.y;
	int z = (int) (blockIdx.z * blockDim.z) + threadIdx.z;

	// Bounds check
	if ( x > ((int)nx - 1) || y > ((int)ny - 1) || z > ((int)nz - 1)){
		return;
	}

	// Get linear index into projection output
	//size_t ind_global = x + (y * (int)nx) + (z * (int)nx * (int)ny);

	// Normalize backprojection
	float weightSum = 0;
	float correction = 0;
	float current = 0;
	correction = back_project(tex_correctiveProjections, hd_sourcePhi, hd_sourceZ, x, y, z, SI, DI, nu, nv, du, dv, nx, ny, nz, ntSubset, &weightSum);

	// Get current value for this voxel
	current = tex3D<float>(tex_volume, (x + 0.5f), (y + 0.5f), (z + 0.5f));

	// Add correction to current value
	if( weightSum != 0){
	current += ((correction / weightSum) * lamda);
	}

	// Write result
	surf3Dwrite(current, surf_correctiveVolume, x * sizeof(float), y, z);
}

// Copy from cudaArray to linear memory (there's probably a better way to do this?)
__global__ void copy_kernel(float* hd_volume, cudaTextureObject_t tex_volume){

	// Get geometry constants from constant memory
	//float SI = hdc_geometry[0];
	//float DI = hdc_geometry[1];
	//float nu = hdc_geometry[2];
	//float nv = hdc_geometry[3];
	//float du = hdc_geometry[4];
	//float dv = hdc_geometry[5];
	float nx = hdc_geometry[6];
	float ny = hdc_geometry[7];
	float nz = hdc_geometry[8];
	//float nt = hdc_geometry[9];

	// Calculate thread index
	int x = (int) (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (int) (blockIdx.y * blockDim.y) + threadIdx.y;
	int z = (int) (blockIdx.z * blockDim.z) + threadIdx.z;

	// Bounds check
	if ( x > ((int)nx - 1) || y > ((int)ny - 1) || z > ((int)nz - 1)){
		return;
	}

	// Get linear index into projection output
	size_t ind_global = x + (y * (int)nx) + (z * (int)nx * (int)ny);
	
	float current = 0;
	current = tex3D<float>(tex_volume, (x + 0.5f), (y + 0.5f), (z + 0.5f));

	// Write
	hd_volume[ind_global] = current;
}


__global__ void set_zero_kernel(cudaSurfaceObject_t surf_volume){

	// Get geometry constants from constant memory
	//float SI = hdc_geometry[0];
	//float DI = hdc_geometry[1];
	//float nu = hdc_geometry[2];
	//float nv = hdc_geometry[3];
	//float du = hdc_geometry[4];
	//float dv = hdc_geometry[5];
	float nx = hdc_geometry[6];
	float ny = hdc_geometry[7];
	float nz = hdc_geometry[8];
	//float nt = hdc_geometry[9];

	// Calculate thread index
	int x = (int) (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (int) (blockIdx.y * blockDim.y) + threadIdx.y;
	int z = (int) (blockIdx.z * blockDim.z) + threadIdx.z;

	// Bounds check
	if ( x > ((int)nx - 1) || y > ((int)ny - 1) || z > ((int)nz - 1)){
		return;
	}

	surf3Dwrite(0.0f, surf_volume, x * sizeof(float), y, z);
}
__global__ void update_kernel(cudaSurfaceObject_t surf_volume, cudaTextureObject_t tex_correctiveVolume){

	// Get geometry constants from constant memory
	//float SI = hdc_geometry[0];
	//float DI = hdc_geometry[1];
	//float nu = hdc_geometry[2];
	//float nv = hdc_geometry[3];
	//float du = hdc_geometry[4];
	//float dv = hdc_geometry[5];
	float nx = hdc_geometry[6];
	float ny = hdc_geometry[7];
	float nz = hdc_geometry[8];
	//float nt = hdc_geometry[9];

	// Calculate thread index
	int x = (int) (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (int) (blockIdx.y * blockDim.y) + threadIdx.y;
	int z = (int) (blockIdx.z * blockDim.z) + threadIdx.z;

	// Bounds check
	if ( x > ((int)nx - 1) || y > ((int)ny - 1) || z > ((int)nz - 1)){
		return;
	}

	float updated = 0;
	updated = tex3D<float>(tex_correctiveVolume, (x + 0.5f), (y + 0.5f), (z + 0.5f));

	if(updated < 0.0f){
		updated = 0.0f;
	}

	surf3Dwrite(updated, surf_volume, x * sizeof(float), y, z);
}



// SART launch function
void dist_sart_launcher(float* hh_volume, float* hh_projections, float* hh_sourcePhi, float* hh_sourceZ, float* hh_SI, float* hh_DI, float* hh_du, float* hh_dv, size_t nu, size_t nv, size_t nx, size_t ny, size_t nz, size_t nt, size_t nBytesTimepointVector, float* hh_geometry, size_t nBytesGeometry, size_t nBytesVolume, float lamda, size_t nSubsets, size_t ntSubset, int nIterations){

// Allocate CUDA array for projection data 
cudaArray* hda_projections;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
struct cudaExtent projectionExtent = make_cudaExtent(nu, nv, nt);

cudaCheck(cudaMalloc3DArray(&hda_projections, &channelDesc, projectionExtent));
cudaPitchedPtr hhp_projections = make_cudaPitchedPtr((void*) hh_projections, nu * sizeof(float), nu, nv);

// Copy projections to device
cudaMemcpy3DParms copyParams = {0};
copyParams.srcPtr = hhp_projections;
copyParams.dstArray = hda_projections;
copyParams.extent = projectionExtent;
copyParams.kind = cudaMemcpyHostToDevice;
cudaCheck(cudaMemcpy3D(&copyParams));

// Create texture object for projections
cudaResourceDesc projectionResDesc;
memset(&projectionResDesc, 0, sizeof(projectionResDesc));
projectionResDesc.resType = cudaResourceTypeArray;
projectionResDesc.res.array.array = hda_projections;

cudaTextureDesc projectionTexDesc;
memset(&projectionTexDesc, 0, sizeof(projectionTexDesc));
projectionTexDesc.addressMode[0] = cudaAddressModeBorder;
projectionTexDesc.addressMode[1] = cudaAddressModeBorder;
projectionTexDesc.addressMode[2] = cudaAddressModeBorder;
projectionTexDesc.filterMode = cudaFilterModeLinear;
projectionTexDesc.readMode = cudaReadModeElementType;
projectionTexDesc.normalizedCoords = 0;

cudaTextureObject_t tex_projections = 0;
cudaCreateTextureObject(&tex_projections, &projectionResDesc, &projectionTexDesc, NULL);

//http://stackoverflow.com/questions/38701467/3d-array-writing-and-reading-as-texture-in-cuda

// Allocate CUDA array for corrective projection data (using 3d even though for now it is 2d -- in the future process more than 1 slice at a time?
cudaArray* hda_correctiveProjections;
struct cudaExtent correctiveExtent = make_cudaExtent(nu, nv, ntSubset);
cudaCheck(cudaMalloc3DArray(&hda_correctiveProjections, &channelDesc, correctiveExtent));

// Texture object for corrective projection reads
cudaTextureDesc correctiveProjectionTexDesc;
memset(&correctiveProjectionTexDesc, 0, sizeof(correctiveProjectionTexDesc));
correctiveProjectionTexDesc.addressMode[0] = cudaAddressModeBorder;
correctiveProjectionTexDesc.addressMode[1] = cudaAddressModeBorder;
correctiveProjectionTexDesc.addressMode[2] = cudaAddressModeBorder;
correctiveProjectionTexDesc.filterMode = cudaFilterModeLinear;
correctiveProjectionTexDesc.readMode = cudaReadModeElementType;
correctiveProjectionTexDesc.normalizedCoords = 0;

cudaResourceDesc correctiveProjectionResDesc;
memset(&correctiveProjectionResDesc, 0, sizeof(correctiveProjectionResDesc));
correctiveProjectionResDesc.resType = cudaResourceTypeArray;
correctiveProjectionResDesc.res.array.array = hda_correctiveProjections;

cudaTextureObject_t tex_correctiveProjections = 0;
cudaCreateTextureObject(&tex_correctiveProjections, &correctiveProjectionResDesc, &correctiveProjectionTexDesc, NULL);

// Surface object for corrective projection writes
cudaSurfaceObject_t surf_correctiveProjections = 0;
cudaCheck(cudaCreateSurfaceObject(&surf_correctiveProjections, &correctiveProjectionResDesc)); 

// Allocate CUDA array for volume
cudaArray* hda_volume;
struct cudaExtent volumeExtent = make_cudaExtent(nx,ny,nz);
cudaCheck(cudaMalloc3DArray(&hda_volume, &channelDesc, volumeExtent));

// Allocate CUDA array for corrective volume
cudaArray* hda_correctiveVolume;
cudaCheck(cudaMalloc3DArray(&hda_correctiveVolume, &channelDesc, volumeExtent));

// Create texture object for volume reads
cudaResourceDesc volumeResDesc;
memset(&volumeResDesc, 0, sizeof(volumeResDesc));
volumeResDesc.resType = cudaResourceTypeArray;
volumeResDesc.res.array.array = hda_volume;

cudaTextureDesc volumeTexDesc;
memset(&volumeTexDesc, 0, sizeof(volumeTexDesc));
volumeTexDesc.addressMode[0] = cudaAddressModeBorder;
volumeTexDesc.addressMode[1] = cudaAddressModeBorder;
volumeTexDesc.addressMode[2] = cudaAddressModeBorder;
volumeTexDesc.filterMode = cudaFilterModeLinear;
volumeTexDesc.readMode = cudaReadModeElementType;
volumeTexDesc.normalizedCoords = 0;

cudaTextureObject_t tex_volume = 0;
cudaCreateTextureObject(&tex_volume, &volumeResDesc, &volumeTexDesc, NULL);

// Create texture object for corrective volume reads
volumeResDesc.res.array.array = hda_correctiveVolume;
cudaTextureObject_t tex_correctiveVolume = 0;
cudaCreateTextureObject(&tex_correctiveVolume, &volumeResDesc, &volumeTexDesc, NULL);

// Create surface object for volume writes
cudaSurfaceObject_t surf_volume = 0;
volumeResDesc.res.array.array = hda_volume;
cudaCreateSurfaceObject(&surf_volume, &volumeResDesc); 

// Create surface object for corrective volume writes
cudaSurfaceObject_t surf_correctiveVolume = 0;
volumeResDesc.res.array.array = hda_correctiveVolume;
cudaCreateSurfaceObject(&surf_correctiveVolume, &volumeResDesc); 

// Allocate global memory for remaining arrays and copy to device
float* hd_sourcePhi;
cudaCheck(cudaMalloc((void**)&hd_sourcePhi, nBytesTimepointVector));
cudaCheck(cudaMemcpy(hd_sourcePhi, hh_sourcePhi, nBytesTimepointVector, cudaMemcpyHostToDevice));

float* hd_sourceZ;
if(hh_sourceZ == NULL){
	hd_sourceZ = NULL;
}
else{
cudaCheck(cudaMalloc((void**)&hd_sourceZ, nBytesTimepointVector));
cudaCheck(cudaMemcpy(hd_sourceZ, hh_sourceZ, nBytesTimepointVector, cudaMemcpyHostToDevice));
}

// Copy geoemtric distances to constant memory
hh_geometry[0] = *hh_SI;
hh_geometry[1] = *hh_DI;
hh_geometry[2] = (float) nu;
hh_geometry[3] = (float) nv; 
hh_geometry[4] = *hh_du;
hh_geometry[5] = *hh_dv;
hh_geometry[6] = (float) nx;
hh_geometry[7] = (float) ny;
hh_geometry[8] = (float) nz;
hh_geometry[9] = (float) nt;

cudaCheck(cudaMemcpyToSymbol(hdc_geometry, hh_geometry, nBytesGeometry));

// Allocate device memory for output image volume
float* hd_volume;
cudaCheck(cudaMalloc((void**)&hd_volume, nBytesVolume));

// Determine grid sizes
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT,BLOCKDEPTH);
// 1 thread per detector per angle
const dim3 forward_gridSize(nu/BLOCKWIDTH + 1, nv/BLOCKWIDTH + 1, ntSubset/BLOCKDEPTH + 1);
// 1 thread per voxel
const dim3 back_gridSize(nx/BLOCKWIDTH + 1, ny/BLOCKWIDTH + 1, nz/BLOCKDEPTH + 1);

int subsetOffset = 0;

// Set to zero
set_zero_kernel<<<back_gridSize, blockSize>>>(surf_volume);

// SART iterations
for(int iIteration = 0; iIteration < (int) nIterations; iIteration++){
for(int iSubset = 0; iSubset < (int) nSubsets; iSubset++){

	subsetOffset = (iSubset * ntSubset);

	// last subset may have fewer projections than the others
	if(iSubset == (nSubsets - 1)){
	

		// Helical?
		if(hd_sourceZ != NULL){
		
		corrective_forward_kernel<<<forward_gridSize,blockSize>>>(surf_correctiveProjections, tex_projections, tex_volume, (hd_sourcePhi + subsetOffset), (hd_sourceZ + subsetOffset), lamda, (nt - (iSubset * ntSubset)), subsetOffset);
		cudaCheck(cudaDeviceSynchronize());
	
		corrective_back_kernel<<<back_gridSize,blockSize>>>(surf_correctiveVolume, tex_volume, tex_correctiveProjections, (hd_sourcePhi + subsetOffset), (hd_sourceZ + subsetOffset), lamda, (nt - (iSubset * ntSubset)));
		cudaCheck(cudaDeviceSynchronize());
		}
		else{
		corrective_forward_kernel<<<forward_gridSize,blockSize>>>(surf_correctiveProjections, tex_projections, tex_volume, (hd_sourcePhi + subsetOffset), hd_sourceZ, lamda, (nt - (iSubset * ntSubset)), subsetOffset);
		cudaCheck(cudaDeviceSynchronize());
	
		corrective_back_kernel<<<back_gridSize,blockSize>>>(surf_correctiveVolume, tex_volume, tex_correctiveProjections, (hd_sourcePhi + subsetOffset), hd_sourceZ, lamda, (nt - (iSubset * ntSubset)));
		cudaCheck(cudaDeviceSynchronize());
		}
	
		}
	
	else{
		// Helical?
		if(hd_sourceZ != NULL){

		corrective_forward_kernel<<<forward_gridSize,blockSize>>>(surf_correctiveProjections, tex_projections, tex_volume, (hd_sourcePhi + subsetOffset), (hd_sourceZ + subsetOffset), lamda, ntSubset, subsetOffset);
		cudaCheck(cudaDeviceSynchronize());
	
		corrective_back_kernel<<<back_gridSize,blockSize>>>(surf_correctiveVolume, tex_volume, tex_correctiveProjections, (hd_sourcePhi + subsetOffset), (hd_sourceZ + subsetOffset), lamda, ntSubset);
		cudaCheck(cudaDeviceSynchronize());
		}
		else{
		corrective_forward_kernel<<<forward_gridSize,blockSize>>>(surf_correctiveProjections, tex_projections, tex_volume, (hd_sourcePhi + subsetOffset), hd_sourceZ, lamda, ntSubset, subsetOffset);
		cudaCheck(cudaDeviceSynchronize());
	
		corrective_back_kernel<<<back_gridSize,blockSize>>>(surf_correctiveVolume, tex_volume, tex_correctiveProjections, (hd_sourcePhi + subsetOffset), hd_sourceZ, lamda, ntSubset);
		cudaCheck(cudaDeviceSynchronize());
		}

	}

	// update volume
	update_kernel<<<back_gridSize, blockSize>>>(surf_volume, tex_correctiveVolume);
}
}

// Copy back to host
copy_kernel<<<back_gridSize, blockSize>>>(hd_volume, tex_volume);
cudaCheck(cudaDeviceSynchronize());
cudaCheck(cudaMemcpy(hh_volume, hd_volume, nBytesVolume, cudaMemcpyDeviceToHost));

// Free allocated memory
cudaDestroyTextureObject(tex_projections);
cudaDestroyTextureObject(tex_correctiveProjections);
cudaDestroyTextureObject(tex_volume);
cudaDestroyTextureObject(tex_correctiveVolume);

cudaDestroySurfaceObject(surf_volume);
cudaDestroySurfaceObject(surf_correctiveProjections);
cudaDestroySurfaceObject(surf_correctiveVolume);

cudaFreeArray(hda_projections);
cudaFreeArray(hda_correctiveProjections);
cudaFreeArray(hda_volume);
cudaFreeArray(hda_correctiveVolume);

cudaFree(hd_volume);
cudaFree(hd_sourcePhi);
cudaFree(hd_sourceZ);

//Reset device for profiling
//cudaDeviceReset();
return;
}
