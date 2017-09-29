#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <math.h>
#include <malloc.h>
#include "mex.h"

// Cuda Error Checking macro
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      mexErrMsgIdAndTxt("MATLAB:cudaError","Error: %s \n In file %s at line %d.\n", cudaGetErrorString(code), file, line);
   }
}


// Absolute value
#define ABS(a) ((a) > 0 ? (a) : -(a))

// Maximum
#define MAX(a, b) ((a) > (b) ? (a) : (b))
// Minimuim
#define MIN(a, b) ((a) > (b) ? (a) : (b))

#define PI 3.14159265359f
#define dx 1.0f
#define dy 1.0f
#define dz 1.0f

#define N_GEOMETRY_CONSTANTS 10

// Thread block sizes
#define BLOCKWIDTH 16
#define BLOCKHEIGHT 16
#define BLOCKDEPTH 4

__device__ void get_central_detector_array_position(float* dd_centralDetectorPosition, float* dd_sourcePosition, float cosPhi, float sinPhi, float SD, float SO, float sourceZ);

// ** Flat detector array
// Get u1 and v1, 2 orthonormal basis vectors describing the detector plane
__device__ void get_detector_basis_vectors_flat(float* dd_sourcePosition, float* u1, float* v1);

// Position of this thread's detector element
__device__ void get_detector_position_flat(float* dd_detectorPosition, float* dd_centralDetectorPosition, float* u1, float* v1, unsigned int u, unsigned int v, float nu, float nv, float du, float dv);

// Boundaries of this thread's detector element
__device__ void get_detector_boundary_positions_flat(float* dd_uF, float* dd_uL, float* dd_vF, float* dd_vL, float* dd_detectorPosition, float* dd_v1, float du, float dv, float cosPhi, float sinPhi);

// ** Source position
__device__ void get_source_position(float* dd_sourcePosition, float cosPhi, float sinPhi, float SI, float sourceZ);

// Decide whether to project detector and voxel boundaries onto XZ or YZ plane
__device__ int get_common_plane(float* dd_sourcePosition, float* dd_detectorPosition);

__device__ float get_slab_intersection_length(float* dd_sourcePosition, float* dd_detectorPosition, int commonPlane, int u, float du, float nu);

// Calculate dx/dy and dz/dy (for XZ plane) or dy/dx and dz/dx ( for YZ plane)
__device__ void get_slopes(float* ka1, float* ka2, float* kz1, float* kz2, float* dd_uF, float* dd_uL, float* dd_vF, float* dd_vL, float* dd_sourcePosition, int commonPlane);

// Find the voxels that are viewed by this detector and their projected boundaries on the common plane
__device__ void get_intersecting_voxels(float* A1, float* A2, float* a1, float* a2, float* Z1, float* Z2, float* z1, float* z2, float ka1, float ka2, float kz1, float kz2, float* dd_sourcePosition, float nx, float ny, float nz, float du, float dv, int commonPlane, int iSlice);

// Calculate the weights for each of the voxels, read their intensity, multipy and sum
__device__ float get_voxel_contributions(float A1, float A2, float a1, float a2, float Z1, float Z2, float z1, float z2, cudaTextureObject_t tex_volume, float nx, float ny, float nz, float slabIntersectionLength, int commonPlane, int iSlice);

__device__ void xyz2uv(float*uv, float* x, float* x0, float* u1, float* v1);

__device__ void get_uv_ind_flat(int* uvInd, float u, float v, float du, float dv, float nu, float nv);

__device__ void get_intersection_line_plane(float* intersection, float* p1, float* p2, float* x0, float* n);

__device__ void get_uv_ranges(float* uMin, float* uMax, float* vMin, float* vMax, float* uv, float* dd_intersection, float* a1, float* a2, float* z1, float* z2, float* dd_sourcePosition, float* dd_centralDetectorPosition, float* u1, float* v1, float* dd_helical_detector_vector);

// forward kernel
__global__ void forward_projection_kernel(float* hd_projections, cudaTextureObject_t tex_volume, float* hd_sourcePhi, float* hd_sourceZ);

//// back kernel
__global__ void back_projection_kernel(float* hd_volume, cudaTextureObject_t tex_projections, float* hd_sourcePhi, float* hd_sourceZ);

// forward launcher
void dist_forward_project(float* hh_projections, float* hh_volume, float* hh_sourcePhi, float* hd_sourceZ, float* hh_SI, float* hh_DI, float* hh_nu, float* hh_nv, float* hh_du, float* hh_dv, size_t nx, size_t ny, size_t nz, size_t nt, size_t nVoxels, float* hh_geometry);

// back launcher
void dist_back_project(float* hh_volume, float* hh_projections, float* hh_sourcePhi, float* hd_sourceZ, float* hh_SI, float* hh_DI, float* hh_du, float* hh_dv, size_t nu, size_t nv, size_t nx, size_t ny, size_t nz, size_t nt, size_t nBytesTimepointVector, float* hh_geometry, size_t nBytesGeometry, size_t nBytesVolume);
 
// SART
 __device__ float get_voxel_contributions(float A1, float A2, float a1, float a2, float Z1, float Z2, float z1, float z2, cudaTextureObject_t tex_volume, float nx, float ny, float nz, float slabIntersectionLength, int commonPlane, int iSlice, float* weightSum);
 
__device__ float forward_project(cudaTextureObject_t tex_volume, float* hd_sourcePhi, float* hd_sourceZ, int u, int v, int t, float SI, float DI, float nu, float nv, float du, float dv, float nx, float ny, float nz, float* weightSum);

__device__ float back_project(cudaSurfaceObject_t surf_volume, cudaTextureObject_t tex_correctiveProjections, float* hd_sourcePhi, float* hd_sourceZ, int x, int y, int z, float SI, float DI, float nu, float nv, float du, float dv, float nx, float ny, float nz, int nt, int normalize);

__global__ void corrective_forward_kernel(cudaSurfaceObject_t surf_correctiveProjections, cudaTextureObject_t tex_projections, cudaTextureObject_t tex_volume, float* hd_sourcePhi, float* hd_sourceZ, float lamda);

__global__ void corrective_back_kernel(cudaSurfaceObject_t surf_volume, cudaTextureObject_t tex_volume, cudaTextureObject_t tex_correctiveProjections, float* hd_sourcePhi, float* hd_sourceZ, float lamda, int ntSubset);

__global__ void copy_kernel(float* hd_volume, cudaTextureObject_t tex_volume);

void dist_sart_launcher(float* hh_volume, float* hh_projections, float* hh_sourcePhi, float* hh_sourceZ, float* hh_SI, float* hh_DI, float* hh_du, float* hh_dv, size_t nu, size_t nv, size_t nx, size_t ny, size_t nz, size_t nt, size_t nBytesTimepointVector, float* hh_geometry, size_t nBytesGeometry, size_t nBytesVolume, float lamda, size_t nSubsets, size_t ntSubset, int nIterations);
void dist_back_project_mag(float* hh_volume, float* hh_projections, float* hh_sourcePhi, float* hh_sourceZ, float* hh_SI, float* hh_DI, float* hh_du, float* hh_dv, size_t nu, size_t nv, size_t nx, size_t ny, size_t nz, size_t nt, size_t nBytesTimepointVector, float* hh_geometry, size_t nBytesGeometry, size_t nBytesVolume);
