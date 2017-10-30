clear; close all;clc
na = 2;
nb = 2;
proj1 = zeros(2,2,'single');
proj = ones(2,2,'single');
for i = 1:10
    host_add(proj1, proj, 0, na, nb);
    proj1
end