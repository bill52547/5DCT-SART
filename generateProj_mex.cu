#include "SART_cuda.h" // consists all required package and functions

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define GEO_PARA prhs[1]
#define ITER_PARA prhs[2]
#define OUT_PROJ plhs[0]

int nx, ny, nz, na, nb, numImg, numBytesImg, numSingleProj, numBytesSingleProj;
float da, db, ai, bi, SO, SD;

// resolutions of volumes 
if (mxGetField(GEO_PARA, 0, "nx") != NULL)
    nx = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nx"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution nx.\n");

if (mxGetField(GEO_PARA, 0, "ny") != NULL)
    ny = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "ny"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution ny.\n");

if (mxGetField(GEO_PARA, 0, "nz") != NULL)
    nz = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nz"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution nz.\n");

numImg = nx * ny * nz; // size of image
numBytesImg = numImg * sizeof(float); // number of bytes in image

// detector plane resolutions
if (mxGetField(GEO_PARA, 0, "na") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "na"));
else if (mxGetField(GEO_PARA, 0, "nv") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nv"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector in plane, which is denoted as na or nu.\n");

if (mxGetField(GEO_PARA, 0, "nb") != NULL)
    nb = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nb"));
else if (mxGetField(GEO_PARA, 0, "nu") != NULL)
    nb = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nu"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector across plane, which is denoted as nb or nv.\n");

numSingleProj = na * nb;
numBytesSingleProj = numSingleProj * sizeof(float);

// detector resolution
if (mxGetField(GEO_PARA, 0, "da") != NULL)
    da = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "da"));
else{
    da = 1.0f;
    mexPrintf("Automatically set detector cell size da to 1. \n");
    mexPrintf("If don't want that default value, please set para.da manually.\n");
}

if (mxGetField(GEO_PARA, 0, "db") != NULL)
    db = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "db"));
else{
    db = 1.0f;
    mexPrintf("Automatically set detectof cell size db to 1. \n");
    mexPrintf("If don't want that default value, please set para.db manually.\n");
}


// detector plane offset from centered calibrations
if (mxGetField(GEO_PARA, 0, "ai") != NULL){
    ai = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "ai"));
    ai -= (float)na / 2 - 0.5f;
}
else{
    mexPrintf("Automatically set detector offset ai to 0. \n");
    mexPrintf("If don't want that default value, please set para.ai manually.\n");
    ai = - (float)na / 2 + 0.5f;
}

if (mxGetField(GEO_PARA, 0, "bi") != NULL){
    bi = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "bi"));
    if (bi > -1)
        bi -= (float)nb / 2 - 0.5f;
}
else{
    mexPrintf("Automatically set detector offset bi to 0. \n");
    mexPrintf("If don't want that default value, please set para.bi manually.\n");
    bi = - (float)nb / 2 + 0.5f;
}


if (mxGetField(GEO_PARA, 0, "SO") != NULL)
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SO"));
else if (mxGetField(GEO_PARA, 0, "SI") != NULL)
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SI"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and isocenter, which is denoted with para.SO or para.DI.\n");

if (mxGetField(GEO_PARA, 0, "SD") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SD"));
else if (mxGetField(GEO_PARA, 0, "DI") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "DI")) + SO;
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and detector plane, which is denoted with para.SD or para.SI + para.DI.\n");


// load iterating parameters, for the whole bin

int numProj, numBytesProj, N_view; // number of bins, numbers of views of bins, and the index view of each bin.
// e.g. we have 3 bins here with 10 total views. For these 3 bins, they holds 1,3,6 views. Then we will set n_views as {0, 1, 4, 10}, which is the starting view indices of each bin. Moreover, we need to well arrange the volumes and flows.


if (mxGetField(ITER_PARA, 0, "N_views") != NULL)
    N_view = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "N_views"));
else{
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number angles, which is denoted as iter_para.N_views.\n");
}

// 5D models
float *h_alpha_x, *h_alpha_y, *h_alpha_z, *h_beta_x, *h_beta_y, *h_beta_z, *angles;

// load 5DCT alpha and beta
if (mxGetField(ITER_PARA, 0, "alpha_x") != NULL)
    h_alpha_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_x")); 
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.alpha_x.\n");    

if (mxGetField(ITER_PARA, 0, "alpha_y") != NULL)
    h_alpha_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_y")); 
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.alpha_y.\n");

if (mxGetField(ITER_PARA, 0, "alpha_z") != NULL)
    h_alpha_z = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_z"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.alpha_z.\n");

if (mxGetField(ITER_PARA, 0, "beta_x") != NULL)
    h_beta_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_x"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.beta_x.\n");

if (mxGetField(ITER_PARA, 0, "beta_y") != NULL)
    h_beta_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_y")); 
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.beta_y.\n");

if (mxGetField(ITER_PARA, 0, "beta_z") != NULL)
    h_beta_z = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_z"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.beta_z.\n");

// load 5DCT parameters volume (v) and flow (f)
float *volumes, *flows;
if (mxGetField(ITER_PARA, 0, "volumes") != NULL)
    volumes= (float*)mxGetData(mxGetField(ITER_PARA, 0, "volumes"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume in iter_para.volumes.\n");  

if (mxGetField(ITER_PARA, 0, "flows") != NULL)
    flows = (float*)mxGetData(mxGetField(ITER_PARA, 0, "flows"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid flow in iter_para.flows.\n");    

if (mxGetField(ITER_PARA, 0, "angles") != NULL)
    angles = (float*)mxGetData(mxGetField(ITER_PARA, 0, "angles"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid angles iter_para.angles.\n");

numProj = numSingleProj * N_view;
numBytesProj = numProj * sizeof(float);

// load initial guess of image
float *h_img;
h_img = (float*)mxGetData(IN_IMG);


// define thread distributions
const dim3 gridSize_img((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (nz + BLOCKDEPTH - 1) / BLOCKDEPTH);
const dim3 gridSize_singleProj((nb + BLOCKWIDTH - 1) / BLOCKWIDTH, (na + BLOCKHEIGHT - 1) / BLOCKHEIGHT, 1);
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT, BLOCKDEPTH);

// CUDA 3DArray Malloc parameters
struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
struct cudaExtent extent_singleProj = make_cudaExtent(nb, na, 1);

//Allocate CUDA array in device memory of 5DCT matrices: alpha and beta
cudaArray *d_alpha_x, *d_alpha_y, *d_alpha_z, *d_beta_x, *d_beta_y, *d_beta_z;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

cudaError_t cudaStat;
// alpha_x
cudaStat = cudaMalloc3DArray(&d_alpha_x, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for alpha_x failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

// alpha_y
cudaStat = cudaMalloc3DArray(&d_alpha_y, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for alpha_y failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

// alpha_z
cudaStat = cudaMalloc3DArray(&d_alpha_z, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for alpha_z failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

// beta_x
cudaStat = cudaMalloc3DArray(&d_beta_x, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for beta_x failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}
// beta_y
cudaStat = cudaMalloc3DArray(&d_beta_y, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for beta_y failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}
// beta_z
cudaStat = cudaMalloc3DArray(&d_beta_z, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for beta_z failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}


// Get pitched pointer to alpha and beta in host memory
cudaPitchedPtr hp_alpha_x = make_cudaPitchedPtr((void*) h_alpha_x, nx * sizeof(float), nx, ny);
cudaPitchedPtr hp_alpha_y = make_cudaPitchedPtr((void*) h_alpha_y, nx * sizeof(float), nx, ny);
cudaPitchedPtr hp_alpha_z = make_cudaPitchedPtr((void*) h_alpha_z, nx * sizeof(float), nx, ny);
cudaPitchedPtr hp_beta_x = make_cudaPitchedPtr((void*) h_beta_x, nx * sizeof(float), nx, ny);
cudaPitchedPtr hp_beta_y = make_cudaPitchedPtr((void*) h_beta_y, nx * sizeof(float), nx, ny);
cudaPitchedPtr hp_beta_z = make_cudaPitchedPtr((void*) h_beta_z, nx * sizeof(float), nx, ny);

// Copy alpha and beta to texture memory from pitched pointer
cudaMemcpy3DParms copyParams = {0};
copyParams.extent = extent_img;
copyParams.kind = cudaMemcpyHostToDevice;

//alpha_x
copyParams.srcPtr = hp_alpha_x;
copyParams.dstArray = d_alpha_x;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy alpha_x to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

//alpha_y
copyParams.srcPtr = hp_alpha_y;
copyParams.dstArray = d_alpha_y;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy alpha_y to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

//alpha_z
copyParams.srcPtr = hp_alpha_z;
copyParams.dstArray = d_alpha_z;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy alpha_z to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

//beta_x
copyParams.srcPtr = hp_beta_x;
copyParams.dstArray = d_beta_x;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy beta_x to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

//beta_y
copyParams.srcPtr = hp_beta_y;
copyParams.dstArray = d_beta_y;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy beta_y to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

//beta_z
copyParams.srcPtr = hp_beta_z;
copyParams.dstArray = d_beta_z;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy beta_z to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}


// create texture object alpha and beta
cudaResourceDesc resDesc;
cudaTextureDesc texDesc, texDesc2;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;

memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;

memset(&texDesc2, 0, sizeof(texDesc2));
texDesc2.addressMode[0] = cudaAddressModeClamp;
texDesc2.addressMode[1] = cudaAddressModeClamp;
texDesc2.addressMode[2] = cudaAddressModeClamp;
texDesc2.filterMode = cudaFilterModePoint;
texDesc2.readMode = cudaReadModeElementType;
texDesc2.normalizedCoords = 0;

// alpha_x
resDesc.res.array.array = d_alpha_x;
cudaTextureObject_t tex_alpha_x = 0;
cudaCreateTextureObject(&tex_alpha_x, &resDesc, &texDesc, NULL);

// alpha_y
resDesc.res.array.array = d_alpha_y;
// memset(&texDesc, 0, sizeof(texDesc));
// texDesc.addressMode[0] = cudaAddressModeClamp;
// texDesc.addressMode[1] = cudaAddressModeClamp;
// texDesc.addressMode[2] = cudaAddressModeClamp;
// texDesc.filterMode = cudaFilterModeLinear;
// texDesc.readMode = cudaReadModeElementType;
// texDesc.normalizedCoords = 0;
cudaTextureObject_t tex_alpha_y = 0;
cudaCreateTextureObject(&tex_alpha_y, &resDesc, &texDesc, NULL);

// alpha_z
resDesc.res.array.array = d_alpha_z;
// memset(&texDesc, 0, sizeof(texDesc));
// texDesc.addressMode[0] = cudaAddressModeClamp;
// texDesc.addressMode[1] = cudaAddressModeClamp;
// texDesc.addressMode[2] = cudaAddressModeClamp;
// texDesc.filterMode = cudaFilterModeLinear;
// texDesc.readMode = cudaReadModeElementType;
// texDesc.normalizedCoords = 0;
cudaTextureObject_t tex_alpha_z = 0;
cudaCreateTextureObject(&tex_alpha_z, &resDesc, &texDesc, NULL);

// beta_x
resDesc.res.array.array = d_beta_x;
// memset(&texDesc, 0, sizeof(texDesc));
// texDesc.addressMode[0] = cudaAddressModeClamp;
// texDesc.addressMode[1] = cudaAddressModeClamp;
// texDesc.addressMode[2] = cudaAddressModeClamp;
// texDesc.filterMode = cudaFilterModeLinear;
// texDesc.readMode = cudaReadModeElementType;
// texDesc.normalizedCoords = 0;
cudaTextureObject_t tex_beta_x = 0;
cudaCreateTextureObject(&tex_beta_x, &resDesc, &texDesc, NULL);

// beta_y
resDesc.res.array.array = d_beta_y;
// memset(&texDesc, 0, sizeof(texDesc));
// texDesc.addressMode[0] = cudaAddressModeClamp;
// texDesc.addressMode[1] = cudaAddressModeClamp;
// texDesc.addressMode[2] = cudaAddressModeClamp;
// texDesc.filterMode = cudaFilterModeLinear;
// texDesc.readMode = cudaReadModeElementType;
// texDesc.normalizedCoords = 0;
cudaTextureObject_t tex_beta_y = 0;
cudaCreateTextureObject(&tex_beta_y, &resDesc, &texDesc, NULL);

// beta_z
resDesc.res.array.array = d_beta_z;
// memset(&texDesc, 0, sizeof(texDesc));
// texDesc.addressMode[0] = cudaAddressModeClamp;
// texDesc.addressMode[1] = cudaAddressModeClamp;
// texDesc.addressMode[2] = cudaAddressModeClamp;
// texDesc.filterMode = cudaFilterModeLinear;
// texDesc.readMode = cudaReadModeElementType;
// texDesc.normalizedCoords = 0;
cudaTextureObject_t tex_beta_z = 0;
cudaCreateTextureObject(&tex_beta_z, &resDesc, &texDesc, NULL);

// malloc in device: projection of the whole bin
float *d_proj;
cudaMalloc((void**)&d_proj, numBytesSingleProj);

// malloc in device: projection of the whole bin
float *d_img ,*d_img1;
cudaArray* array_img;
cudaMalloc((void**)&d_img, numBytesImg);
cudaMalloc((void**)&d_img1, numBytesImg);
cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);
cudaStat = cudaMalloc3DArray(&array_img, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for array_img failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

float angle, volume, flow;

//Malloc forward and inverted DVFs in device
float *d_mx, *d_my, *d_mz, *d_mx2, *d_my2, *d_mz2;
cudaMalloc(&d_mx, numBytesImg);
cudaMalloc(&d_my, numBytesImg);
cudaMalloc(&d_mz, numBytesImg);
cudaMalloc(&d_mx2, numBytesImg);
cudaMalloc(&d_my2, numBytesImg);
cudaMalloc(&d_mz2, numBytesImg);


// Alloc forward and inverted DVFs in device, in form of array memory
cudaArray *array_mx, *array_my, *array_mz, *array_mx2, *array_my2, *array_mz2;
cudaStat = cudaMalloc3DArray(&array_mx, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for array_mx failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

cudaStat = cudaMalloc3DArray(&array_my, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for array_my failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

cudaStat = cudaMalloc3DArray(&array_mz, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for array_mz failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

cudaStat = cudaMalloc3DArray(&array_mx2, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for array_mx2 failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

cudaStat = cudaMalloc3DArray(&array_my2, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for array_my2 failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

cudaStat = cudaMalloc3DArray(&array_mz2, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for array_mz2 failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

// define tex_mx etc
cudaTextureObject_t tex_mx = 0, tex_my = 0, tex_mz = 0, tex_mx2 = 0, tex_my2 = 0, tex_mz2 = 0, tex_img = 0;


// setup output images
OUT_PROJ = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
const mwSize outDim[3] = {(mwSize)nb, (mwSize)na, (mwSize)N_view};

mxSetDimensions(OUT_PROJ, outDim, 3);
mxSetData(OUT_PROJ, mxMalloc(numBytesProj));
float *h_outproj = (float*)mxGetData(OUT_PROJ);
copyParams.kind = cudaMemcpyDeviceToDevice;

cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) d_img, nx * sizeof(float), nx, ny);
copyParams.srcPtr = dp_img;
copyParams.dstArray = array_img;
cudaStat = cudaMemcpy3D(&copyParams);   
if (cudaStat != cudaSuccess) {
    mexPrintf("Failed to copy dp_img to array memory array_img.\n");
    mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}
resDesc.res.array.array = array_img;
cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);


for (int i_view = 0; i_view < N_view; i_view++)
{
    mexPrintf("Projecting %d view, of all %d views.\n", i_view, N_view); mexEvalString("drawnow;");
    volume = volumes[i_view];
    flow = flows[i_view];
    angle = angles[i_view];

    kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, tex_alpha_x, tex_alpha_y, tex_alpha_z, tex_beta_x, tex_beta_y, tex_beta_z, volume, flow, nx, ny, nz);
    cudaDeviceSynchronize();

    // copy mx etc to pitched pointer and bind it to a texture object
    cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void*) d_mx, nx * sizeof(float), nx, ny);
    copyParams.srcPtr = dp_mx;
    copyParams.dstArray = array_mx;
    cudaStat = cudaMemcpy3D(&copyParams);   
    if (cudaStat != cudaSuccess) {
        mexPrintf("Failed to copy dp_mx to array memory array_mx2.\n");
        mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
            mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
    }
    resDesc.res.array.array = array_mx;
    cudaCreateTextureObject(&tex_mx, &resDesc, &texDesc, NULL);

    cudaPitchedPtr dp_my = make_cudaPitchedPtr((void*) d_my, nx * sizeof(float), nx, ny);
    copyParams.srcPtr = dp_my;
    copyParams.dstArray = array_my;
    cudaStat = cudaMemcpy3D(&copyParams);   
    if (cudaStat != cudaSuccess) {
        mexPrintf("Failed to copy dp_my to array memory array_mx2.\n");
        mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
            mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
    }
    resDesc.res.array.array = array_my;
    cudaCreateTextureObject(&tex_my, &resDesc, &texDesc, NULL);

    cudaPitchedPtr dp_mz = make_cudaPitchedPtr((void*) d_mz, nx * sizeof(float), nx, ny);
    copyParams.srcPtr = dp_mz;
    copyParams.dstArray = array_mz;
    cudaStat = cudaMemcpy3D(&copyParams);   
    if (cudaStat != cudaSuccess) {
        mexPrintf("Failed to copy dp_mz to array memory array_mx2.\n");
        mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
            mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
    }
    resDesc.res.array.array = array_mz;
    cudaCreateTextureObject(&tex_mz, &resDesc, &texDesc, NULL);

    kernel_invertDVF<<<gridSize_img, blockSize>>>(d_mx2, d_my2, d_mz2, tex_mx, tex_my, tex_mz, nx, ny, nz, 10);
    cudaDeviceSynchronize();        
            
    kernel_deformation<<<gridSize_img, blockSize>>>(d_img1, tex_img, d_mx2, d_my2, d_mz2, nx, ny, nz);
    cudaDeviceSynchronize();

    kernel_projection<<<gridSize_singleProj, blockSize>>>(d_proj, d_img1, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaMemcpy(h_outproj + i_view * numSingleProj, d_proj, numBytesSingleProj, cudaMemcpyDeviceToHost);
}


cudaDestroyTextureObject(tex_alpha_x);
cudaDestroyTextureObject(tex_alpha_y);
cudaDestroyTextureObject(tex_alpha_z);
cudaDestroyTextureObject(tex_beta_x);
cudaDestroyTextureObject(tex_beta_y);
cudaDestroyTextureObject(tex_beta_z);
cudaDestroyTextureObject(tex_img);
cudaDestroyTextureObject(tex_mx);
cudaDestroyTextureObject(tex_my);
cudaDestroyTextureObject(tex_mz);
cudaDestroyTextureObject(tex_mx2);
cudaDestroyTextureObject(tex_my2);
cudaDestroyTextureObject(tex_mz2);

cudaFreeArray(d_alpha_x);
cudaFreeArray(d_alpha_y);
cudaFreeArray(d_alpha_z);
cudaFreeArray(d_beta_x);
cudaFreeArray(d_beta_y);
cudaFreeArray(d_beta_z);
// cudaFreeArray(d_img);
cudaFree(d_mx);
cudaFree(d_my);
cudaFree(d_mz);
cudaFree(d_mx2);
cudaFree(d_my2);
cudaFree(d_mz2);
cudaFreeArray(array_mx);
cudaFreeArray(array_my);
cudaFreeArray(array_mz);
cudaFreeArray(array_mx2);
cudaFreeArray(array_my2);
cudaFreeArray(array_mz2);
cudaFree(d_proj);

cudaFree(d_img);
cudaFree(d_img1);
cudaDeviceReset();
return;
}

