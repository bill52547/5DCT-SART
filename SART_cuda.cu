#include "SART_cuda.h" // consists all required package and functions

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define PROJ prhs[1]
#define GEO_PARA prhs[2]
#define ITER_PARA prhs[3]
#define OUT_IMG plhs[0]

// load parameters
// assume all the parameter are orginized as:
// dx = dy = dz = 1 
// da = db

// load geometry parameters, all need parameter for single view projection
int nx, ny, nz, na, nb, numImg, numBytesImg, numSingleProj, numBytesSingleProj;
float da, db, ai, bi, SO, SD, dx;

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
else if (mxGetField(GEO_PARA, 0, "nu") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nu"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector in plane, which is denoted as na or nu.\n");

if (mxGetField(GEO_PARA, 0, "nb") != NULL)
    nb = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nb"));
else if (mxGetField(GEO_PARA, 0, "nv") != NULL)
    nb = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nv"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector across plane, which is denoted as nb or nv.\n");

numSingleProj = na * nb;
numBytesSingleProj = numSingleProj * sizeof(float);
if (mxGetField(GEO_PARA, 0, "dx") != NULL)
    dx = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "dx"));
else{
    dx = 1;
    mexPrintf("Automatically set voxel size dx to 1. \n");
    mexPrintf("If don't want that default value, please set para.dx manually.\n");
}
if (mxGetField(GEO_PARA, 0, "da") != NULL)
    da = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "da")) / dx;
else{
    da = 1.0f / dx;
    mexPrintf("Automatically set detector cell size da to 1. \n");
    mexPrintf("If don't want that default value, please set para.da manually.\n");
}
if (mxGetField(GEO_PARA, 0, "db") != NULL)
    db = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "db")) / dx;
else{
    db = 1.0f / dx;
    mexPrintf("Automatically set detectof cell size db to 1. \n");
    mexPrintf("If don't want that default value, please set para.db manually.\n");
}


// detector plane offset from centered calibrations
if (mxGetField(GEO_PARA, 0, "ai") != NULL){
    ai = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "ai"));
    if (ai > -1)
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
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SO")) / dx;
else if (mxGetField(GEO_PARA, 0, "DI") != NULL)
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "DI")) / dx;
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and isocenter, which is denoted with para.SO or para.DI.\n");
if (mxGetField(GEO_PARA, 0, "SD") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SD")) / dx;
else if (mxGetField(GEO_PARA, 0, "SI") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SI")) / dx + SO;
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and detector plane, which is denoted with para.SD or para.SI + para.DI.\n");

// load iterating parameters, for the whole bin
int n_view, n_iter, numProj, numBytesProj;
float *h_alpha_x, *h_alpha_y, *h_alpha_z, *h_beta_x, *h_beta_y, *h_beta_z, *angles, lambda;
if (mxGetField(ITER_PARA, 0, "nv") != NULL)
    n_view = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "nv")); // number of views in this bin
else{
    n_view = 1;
    mexPrintf("Automatically set number of views to 1. \n");
    mexPrintf("If don't want that default value, please set iter_para.nv manually.\n");
}
if (mxGetField(ITER_PARA, 0, "n_iter") != NULL)
    n_iter = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_iter")); // number of views in this bin
else{
    n_iter = 1;
    mexPrintf("Automatically set number of iterations to 1. \n");
    mexPrintf("If don't want that default value, please set iter_para.n_iter manually.\n");
}

// load 5DCT alpha and beta
if (mxGetField(ITER_PARA, 0, "alpha_x") != NULL)
    h_alpha_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_x")); 
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.alpha_x.\n");    

// check validation of h_alpha_x
mwSize sizeCheck;
mxClassID classCheck;
const mwSize *dimCheck;
classCheck = mxGetClassID(mxGetField(ITER_PARA, 0, "alpha_x"));
if (classCheck != mxSINGLE_CLASS){
mexErrMsgIdAndTxt("MATLAB:badInput","imageIn, Xi, Yi, and Zi must be of data type single.\n");
}
sizeCheck = mxGetNumberOfDimensions(mxGetField(ITER_PARA, 0, "alpha_x"));
if (sizeCheck != 3){
mexErrMsgIdAndTxt("MATLAB:badInput","imageIn, Xi, Yi, and Zi must be 3D matrices.\n");
}


dimCheck = mxGetDimensions(mxGetField(ITER_PARA, 0, "alpha_x"));
if(nx != dimCheck[0]){
mexErrMsgIdAndTxt("MATLAB:badInput","Xi, Yi, and Zi must be the same size, nx.\n");
}
if(ny != dimCheck[1]){
mexErrMsgIdAndTxt("MATLAB:badInput","Xi, Yi, and Zi must be the same size, ny.\n");
}
if(nz != dimCheck[2]){
mexErrMsgIdAndTxt("MATLAB:badInput","Xi, Yi, and Zi must be the same size, nz.\n");
}

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

numProj = numSingleProj * n_view;
numBytesProj = numProj * sizeof(float);
if (mxGetField(ITER_PARA, 0, "angles") != NULL)
    angles = (float*)mxGetData(mxGetField(ITER_PARA, 0, "angles"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid angles iter_para.angles.\n");
if (mxGetField(ITER_PARA, 0, "lambda") != NULL)
    lambda = (float)mxGetScalar(mxGetField(ITER_PARA, 0, "lambda"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid coefficience iter_para.lambda.\n");

// load initial guess of image
float *h_img;
h_img = (float*)mxGetData(IN_IMG);

// load true projection value
float *h_proj;
h_proj = (float*)mxGetData(PROJ);

// define thread distributions
const dim3 gridSize_img((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (nz + BLOCKDEPTH - 1) / BLOCKDEPTH);
const dim3 gridSize_singleProj((na + BLOCKWIDTH - 1) / BLOCKWIDTH, (nb + BLOCKHEIGHT - 1) / BLOCKHEIGHT, 1);
const dim3 gridSize_proj((na + BLOCKWIDTH - 1) / BLOCKWIDTH, (nb + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (n_view + BLOCKDEPTH - 1) / BLOCKDEPTH);
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT, BLOCKDEPTH);

// CUDA 3DArray Malloc parameters
struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
struct cudaExtent extent_proj = make_cudaExtent(na, nb, n_view);
struct cudaExtent extent_singleProj = make_cudaExtent(na, nb, 1);

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
cudaTextureDesc texDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;

// alpha_x
resDesc.res.array.array = d_alpha_x;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;
cudaTextureObject_t tex_alpha_x = 0;
cudaCreateTextureObject(&tex_alpha_x, &resDesc, &texDesc, NULL);

// alpha_y
resDesc.res.array.array = d_alpha_y;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;
cudaTextureObject_t tex_alpha_y = 0;
cudaCreateTextureObject(&tex_alpha_y, &resDesc, &texDesc, NULL);

// alpha_z
resDesc.res.array.array = d_alpha_z;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;
cudaTextureObject_t tex_alpha_z = 0;
cudaCreateTextureObject(&tex_alpha_z, &resDesc, &texDesc, NULL);

// beta_x
resDesc.res.array.array = d_beta_x;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;
cudaTextureObject_t tex_beta_x = 0;
cudaCreateTextureObject(&tex_beta_x, &resDesc, &texDesc, NULL);

// beta_y
resDesc.res.array.array = d_beta_y;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;
cudaTextureObject_t tex_beta_y = 0;
cudaCreateTextureObject(&tex_beta_y, &resDesc, &texDesc, NULL);

// beta_z
resDesc.res.array.array = d_beta_z;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;
cudaTextureObject_t tex_beta_z = 0;
cudaCreateTextureObject(&tex_beta_z, &resDesc, &texDesc, NULL);

// malloc in device: projection of the whole bin
float *d_proj;
cudaMalloc((void**)&d_proj, numBytesProj);

// copy to device: projection of the whole bin
cudaMemcpy(d_proj, h_proj, numBytesProj, cudaMemcpyHostToDevice);

// malloc in device: another projection pointer, with single view size
float *d_singleViewProj2;
cudaMalloc((void**)&d_singleViewProj2, numBytesSingleProj);

// malloc in device: projection of the whole bin
float *d_img;
cudaMalloc((void**)&d_img, numBytesImg);

// copy to device: initial guess of image
cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);

// malloc in device: another image pointer, for single view 
float *d_singleViewImg1, *d_singleViewImg2, *d_imgOnes;
cudaMalloc(&d_singleViewImg1, numBytesImg);
cudaMalloc(&d_singleViewImg2, numBytesImg);
cudaMalloc(&d_imgOnes, numBytesImg);
float angle, volume, flow;

// Malloc forward and inverted DVFs in device
float *d_mx, *d_my, *d_mz, *d_mx2, *d_my2, *d_mz2;
cudaMalloc(&d_mx, numBytesImg);
cudaMalloc(&d_my, numBytesImg);
cudaMalloc(&d_mz, numBytesImg);
cudaMalloc(&d_mx2, numBytesImg);
cudaMalloc(&d_my2, numBytesImg);
cudaMalloc(&d_mz2, numBytesImg);
float fref = -0.4193;
float vref = 0;
for (int iter = 0; iter < n_iter; iter++){ // iteration
    for (int i_view = 0; i_view < n_view; i_view++){ // view
        // mexPrintf("i_view = %d.\n", i_view);        
        angle = angles[i_view];
        volume = vref - volumes[i_view];
        flow = fref - flows[i_view];

        // generate forwards DVFs: d_mx, d_my, d_mz
        kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, tex_alpha_x, tex_alpha_y, tex_alpha_z, tex_beta_x, tex_beta_y, tex_beta_z, volume, flow, nx, ny, nz);
        cudaDeviceSynchronize();
        // cudaStat = cudaPeekAtLastError();
        // if (cudaStat != cudaSuccess) {
        //     mexPrintf("forward DVF generator kernel launch failure.\n");
        //     mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        //         mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
        // }
        // cudaStat = cudaDeviceSynchronize();
        // if (cudaStat != cudaSuccess) {
        //     mexPrintf("Error in forward DVf generator kernel.\n");
        //     mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        //         mexErrMsgIdAndTxt("MATLAB:cudaFail","Projecton failed.\n");
        // }
        // generate inverted DVFs: d_mx2, d_my2, d_mz2
        kernel_invertDVF<<<gridSize_img, blockSize>>>(d_mx2, d_my2, d_mz2, tex_alpha_x, tex_alpha_y, tex_alpha_z, tex_beta_x, tex_beta_y, tex_beta_z, volume, flow, nx, ny, nz, 10);
        cudaDeviceSynchronize();

        // cudaStat = cudaPeekAtLastError();
        // if (cudaStat != cudaSuccess) {
        //     mexPrintf("inverted DVF generator kernel launch failure.\n");
        //     mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        //         mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
        // }
        // cudaStat = cudaDeviceSynchronize();
        // if (cudaStat != cudaSuccess) {
        //     mexPrintf("Error in inverted DVf generator kernel.\n");
        //     mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        //         mexErrMsgIdAndTxt("MATLAB:cudaFail","Projecton failed.\n");
        // }

        // deformed image for i_view, from reference image of the bin

        kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg1, d_img, d_mx2, d_my2, d_mz2, nx, ny, nz);
        cudaDeviceSynchronize();
        // continue;

        // projection of deformed image from initial guess
        kernel_projection<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_singleViewImg1, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz); // TBD
        cudaDeviceSynchronize();

        // difference between true projection and projection from initial guess
        // update d_singleViewProj2 instead of malloc a new one
        kernel_add<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_proj, i_view, na, nb, -1);
        cudaDeviceSynchronize();
        
        // backprojecting the difference of projections
        // print parameters              
        kernel_backprojection<<<gridSize_img, blockSize>>>(d_singleViewImg1, d_singleViewProj2, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
        cudaDeviceSynchronize();

        // deform backprojection back to the bin
        kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg2, d_singleViewImg1, d_mx, d_my, d_mz, nx, ny, nz);
        cudaDeviceSynchronize();
        // continue;
        // calculate the ones backprojection data
        kernel_initial<<<gridSize_img, blockSize>>>(d_singleViewImg1, nx, ny, nz, 1);
        cudaDeviceSynchronize();

        kernel_projection<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_singleViewImg1, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
        cudaDeviceSynchronize();

        kernel_backprojection<<<gridSize_img, blockSize>>>(d_singleViewImg1, d_singleViewProj2, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
        cudaDeviceSynchronize();

        // weighting
        kernel_division<<<gridSize_img, blockSize>>>(d_singleViewImg2, d_singleViewImg1, nx, ny, nz);
        cudaDeviceSynchronize();
        
        // updating
        kernel_update<<<gridSize_img, blockSize>>>(d_img, d_singleViewImg2, nx, ny, nz, lambda);
        cudaDeviceSynchronize();              
    }
}
OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);



// const mwSize *outDim = mxGetDimensions(PROJ); // IN_IMG or PROJ
// mxSetDimensions(OUT_IMG, outDim, 3);
// mxSetData(OUT_IMG, mxMalloc(numBytesImg));
// float *h_outimg = (float*)mxGetData(OUT_IMG);
// cudaMemcpy(h_outimg, d_singleViewProj2, numBytesSingleProj, cudaMemcpyDeviceToHost);


// const mwSize *outDim = mxGetDimensions(IN_IMG); // IN_IMG or PROJ
// mxSetDimensions(OUT_IMG, outDim, 3);
// mxSetData(OUT_IMG, mxMalloc(numBytesImg));
// float *h_outimg = (float*)mxGetData(OUT_IMG);
// cudaMemcpy(h_outimg, d_singleViewImg1, numBytesImg, cudaMemcpyDeviceToHost);


//archive
const mwSize *outDim = mxGetDimensions(IN_IMG); // IN_IMG or PROJ
mxSetDimensions(OUT_IMG, outDim, 3);
mxSetData(OUT_IMG, mxMalloc(numBytesImg));
float *h_outimg = (float*)mxGetData(OUT_IMG);
cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);

cudaDestroyTextureObject(tex_alpha_x);
cudaDestroyTextureObject(tex_alpha_y);
cudaDestroyTextureObject(tex_alpha_z);
cudaDestroyTextureObject(tex_beta_x);
cudaDestroyTextureObject(tex_beta_y);
cudaDestroyTextureObject(tex_beta_z);

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
cudaFree(d_proj);
cudaFree(d_singleViewImg1);
cudaFree(d_singleViewImg2);
cudaFree(d_singleViewProj2);

cudaFree(d_img);
cudaDeviceReset();
return;
}

