clear; close all;clc
nx = 2;
ny = 2;
nz = 1;
img1 =  ones(nx, ny, nz ,'single');
img = 2 * ones(nx, ny, nz ,'single');
img(1) = 0;
host_division(img1, img, nx, ny, nz);
img1