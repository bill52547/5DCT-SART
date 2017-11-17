#!/bin/bash
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled kernel_add.cu -I/usr/local/MATLAB/R2017b/extern/include 
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled kernel_backprojection.cu -I/usr/local/MATLAB/R2017b/extern/include 
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled kernel_backprojection_pd.cu -I/usr/local/MATLAB/R2017b/extern/include 
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled kernel_forwardDVF.cu -I/usr/local/MATLAB/R2017b/extern/include 
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled kernel_deformation.cu -I/usr/local/MATLAB/R2017b/extern/include 
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled kernel_division.cu -I/usr/local/MATLAB/R2017b/extern/include 
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled kernel_initial.cu -I/usr/local/MATLAB/R2017b/extern/include 
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled kernel_invertDVF.cu -I/usr/local/MATLAB/R2017b/extern/include 
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled kernel_projection.cu -I/usr/local/MATLAB/R2017b/extern/include 
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled kernel_projection_rd.cu -I/usr/local/MATLAB/R2017b/extern/include 
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled kernel_update.cu -I/usr/local/MATLAB/R2017b/extern/include 
nvcc -shared -c -Xcompiler -fPIC -odir ../compiled processBar.cu -I/usr/local/MATLAB/R2017b/extern/include 


