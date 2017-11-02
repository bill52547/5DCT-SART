__global__ void kernel_add(float *proj1, float *proj, int iv, int na, int nb, float weight){
    int ia = 16 * blockIdx.x + threadIdx.x;
    int ib = 16 * blockIdx.y + threadIdx.y;
    if (ia >= na || ib >= nb)
        return;
    proj1[ia + ib * na] += proj[ia + ib * na + iv * na * nb] * weight;
}

// __global__ void kernel_add(cudaArray *proj1, cudaArray *proj, int iv, int na, int nb, float weight){
//     int ia = 16 * blockIdx.x + threadIdx.x;
//     int ib = 16 * blockIdx.y + threadIdx.y;
//     if (ia >= na || ib >= nb)
//         return;
//     proj1[ia + ib * na] += proj[ia + ib * na + iv * na * nb] * weight;
// }
