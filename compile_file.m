close all;clc

!nvcc -shared -c -Xcompiler -fPIC kernel_backprojection.cu -I/usr/local/MATLAB/R2017b/extern/include 
    
if 0
    !nvcc -shared -c -Xcompiler -fPIC kernel_add.cu -I/usr/local/MATLAB/R2017b/extern/include 
    !nvcc -shared -c -Xcompiler -fPIC kernel_deformation.cu -I/usr/local/MATLAB/R2017b/extern/include 
    !nvcc -shared -c -Xcompiler -fPIC kernel_backprojection.cu -I/usr/local/MATLAB/R2017b/extern/include 
    !nvcc -shared -c -Xcompiler -fPIC kernel_division.cu -I/usr/local/MATLAB/R2017b/extern/include 
    !nvcc -shared -c -Xcompiler -fPIC kernel_initial.cu -I/usr/local/MATLAB/R2017b/extern/include 
    !nvcc -shared -c -Xcompiler -fPIC kernel_projection.cu -I/usr/local/MATLAB/R2017b/extern/include 
    !nvcc -shared -c -Xcompiler -fPIC kernel_update.cu -I/usr/local/MATLAB/R2017b/extern/include 
    !nvcc -shared -c -Xcompiler -fPIC kernel_forwardDVF.cu -I/usr/local/MATLAB/R2017b/extern/include 
    !nvcc -shared -c -Xcompiler -fPIC kernel_invertDVF.cu -I/usr/local/MATLAB/R2017b/extern/include 
end

if 0
    mexcuda SART_cuda.cu kernel_add.o kernel_backprojection.o kernel_deformation.o kernel_division.o kernel_initial.o...
    kernel_projection.o kernel_update.o kernel_forwardDVF.o kernel_invertDVF.o
end

if 0
    mexcuda generateProj_mex.cu kernel_deformation.o kernel_projection.o kernel_forwardDVF.o kernel_invertDVF.o
end

if 0
    mexcuda host_projection.cu kernel_projection.o
end

if 01
    mexcuda host_backprojection.cu kernel_backprojection.o
end