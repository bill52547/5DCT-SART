clear; close all;clc

para.na = 512;
para.nb = 512;
para.SO = 1000;
para.SD = 1500;
para.angle = 0.15;
para.ai = -para.na / 2 + 0.5;
para.da = 1;
para.bi = -para.nb / 2 + 0.5;
para.db = 1;
para.nx = 206;
para.ny = 206;
para.nz = 96;
load /home/minghao/Workspace/TestArea/temp5/test_SART_single.mat
para.angle = iter_para.angles(1)

load('/home/minghao/Workspace/TestArea/temp5/phantom');
Img = XCAT(:,:,:,1);
Proj = zeros(para.na, para.nb, 'single');
host_projection(Proj, Img, para);
load /home/minghao/Workspace/TestArea/temp5/test_SART_single.mat proj
Proj1 = proj(:,:,1);

imshow([Proj, Proj1],[])