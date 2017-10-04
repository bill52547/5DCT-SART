% #define IN_IMG prhs[0]
% #define PROJ prhs[1]
% #define GEO_PARA prhs[2]
% #define ITER_PARA prhs[3]
% #define OUT_IMG plhs[0]
% 
% // load parameters
% // assume all the parameter are orginized as:
% // dx = dy = dz = 1 
% // da = db
% 
% // load geometry parameters, all need parameter for single view projection
% int nx, ny, nz, na, nb, numImg, numBytesImg, numSingleProj, numBytesSingleProj;
% float da, db, ai, bi, SO, SD;
% nx = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nx"));
% ny = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "ny"));
% nz = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nz"));
% numImg = nx * ny * nz; // size of image
% numBytesImg = numImg * sizeof(float); // number of bytes in image
% na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "na"));
% nb = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nb"));
% numSingleProj = na * nb;
% numBytesSingleProj = numSingleProj * sizeof(float);
% da = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "da"));
% db = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "db"));
% ai = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "ai"));
% bi = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "bi"));
% SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SO"));
% SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SD"));
% 
% // load iterating parameters, for the whole bin
% int n_view, n_iter, numProj, numBytesProj;
% float *h_mx, *h_my, *h_mz, *h_mx2, *h_my2, *h_mz2, *angles, lambda;
% n_view = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "nv")); // number of views in this bin
% n_iter = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_iter")); // number of iterations of SART
% h_mx = (float*)mxGetData(mxGetField(ITER_PARA, 0, "mx")); // index stead of difference
% h_my = (float*)mxGetData(mxGetField(ITER_PARA, 0, "my")); 
% h_mz = (float*)mxGetData(mxGetField(ITER_PARA, 0, "mz"));
% h_mx2 = (float*)mxGetData(mxGetField(ITER_PARA, 0, "mx2")); // index stead of difference
% h_my2 = (float*)mxGetData(mxGetField(ITER_PARA, 0, "my2")); 
% h_mz2 = (float*)mxGetData(mxGetField(ITER_PARA, 0, "mz2"));
% numProj = numSingleProj * n_view;
% numBytesProj = numProj * sizeof(float);
% angles = (float*)mxGetData(mxGetField(ITER_PARA, 0, "angles"));
% lambda = (float)mxGetScalar(mxGetField(ITER_PARA, 0, "lambda"));
% 
% // load initial guess of image
% float *h_img;
% h_img = (float*)mxGetData(IN_IMG);
% 
% // load true projection value
% float *h_proj;
% h_proj = (float*)mxGetData(PROJ);