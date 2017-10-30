clear; close all;clc

para.na = 512;
para.nb = 512;
para.SO = 500;
para.SD = 1000;
para.angle = 0;
para.ai = -para.na / 2 + 0.5;
para.da = 1;
para.bi = -para.nb / 2 + 0.5;
para.db = 1;
para.nx = 256;
para.ny = 256;
para.nz = 100;

Img = zeros(para.nx, para.ny, para.nz);
for i = 1 : para.nz
    Img(:,:,i) = phantom(para.nx);
end
Img = single(Img);
Proj = zeros(para.na, para.nb, 'single');
host_projection(Proj, Img, para);
Img1 = Img;
host_backprojection(Img1, Proj, para);
view3dgui(Img1);