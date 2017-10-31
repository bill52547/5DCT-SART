close all;clc
mexcuda SART_cuda.cu kernel_add.cu kernel_backprojection.cu kernel_deformation.cu kernel_division.cu kernel_initial.cu...
kernel_projection.cu kernel_update.cu kernel_forwardDVF.cu kernel_invertDVF.cu
% mexcuda host_backprojection.cu kernel_backprojection.cu