__global__ void kernel_add(cudaArray *proj1, cudaArray * proj, int iv, int na, int nb, float weight){
    int ia = blockSize.x * blockIdx.x + threadIdx.x;
    int ib = blockSize.y * blockIdx.y + threadIdx.y;
    if (ia >= na || ib >= nb)
        return;
    proj1[ia][ib][0] += proj[ia][ib][iv] * weight;
}
