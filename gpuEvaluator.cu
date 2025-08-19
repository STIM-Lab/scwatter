#include "gpuEvaluator.h"
#include "tira/cuda/error.h"


float* gpu_z_layers;
int* gpu_waves_begin;
int* gpu_waves_end;
thrust::complex<float>* gpu_W;

//extern float z_up, z_bo;
extern size_t free_gpu_memory;
extern size_t total_gpu_memory;

/// Copy all unpacked plane wave data to the GPU
void gpu_initialize(){
    HANDLE_ERROR(cudaMalloc(&gpu_z_layers, layers * sizeof(float)));                                         // copy z coordinates of layers
    HANDLE_ERROR(cudaMemcpy(gpu_z_layers, &z_layers[0], layers * sizeof(float), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc(&gpu_waves_begin, waves_begin.size() * sizeof(int)));                          // copy the start/end waves for each layer
    HANDLE_ERROR(cudaMemcpy(gpu_waves_begin, &waves_begin[0], waves_begin.size() * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc(&gpu_waves_end, waves_end.size() * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(gpu_waves_end, &waves_end[0], waves_end.size() * sizeof(int), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc(&gpu_W, W.size() * sizeof(UnpackedWave<float>)));                                // copy the waves
    HANDLE_ERROR(cudaMemcpy(gpu_W, &W[0], W.size() * sizeof(UnpackedWave<float>), cudaMemcpyHostToDevice));
}

__device__ void evaluate(thrust::complex<float>& Ex, thrust::complex<float>& Ey, thrust::complex<float>& Ez,
    float x, float y, float z,
    thrust::complex<float>* W, float* z_layers, int* waves_begin, int* waves_end, int layers) {
    // find the current layer
    size_t l = 0;
    for (size_t li = 0; li < layers; li++) {
        if (z >= z_layers[li] - 0.001) {
            l = li + 1;
        }
    }

    size_t size_W = 6;
    size_t begin = waves_begin[l];
    size_t end = waves_end[l];

    thrust::complex<float> kx, ky, kz;
    thrust::complex<float> E0x, E0y, E0z;
    thrust::complex<float> k_dot_r;
    thrust::complex<float> phase;
    thrust::complex<float> i(0.0, 1.0);

    Ex = 0;
    Ey = 0;
    Ez = 0;
    
    for (int idx = begin; idx < end; idx++) {
        E0x = W[idx * size_W + 0];                                      // load the plane wave E vector
        E0y = W[idx * size_W + 1];
        E0z = W[idx * size_W + 2];
        kx = W[idx * size_W + 3];                                       // load the plane wave k vector
        ky = W[idx * size_W + 4];
        kz = W[idx * size_W + 5];

        k_dot_r = kx * x + ky * y + kz * z;
        phase = thrust::exp(i * k_dot_r);
        Ex += E0x * phase;
        Ey += E0y * phase;
        Ez += E0z * phase;
    }

}

__global__ void kernel_xy(thrust::complex<float>* E_xy, thrust::complex<float>* W, float* z_layers, int* waves_begin, int* waves_end,
    float x_start, float y_start, float z, float d, int N, int layers) {
    unsigned int xi = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yi = blockDim.y * blockIdx.y + threadIdx.y;
    if(xi >= N || yi >= N) return;

    unsigned int idx_E = yi * N * 3 + xi * 3;
    
    float x = x_start + d * xi;
    float y = y_start + d * yi;
    
    thrust::complex<float> Ex, Ey, Ez;

    evaluate(Ex, Ey, Ez, x, y, z, W, z_layers, waves_begin, waves_end, layers);

    
    E_xy[idx_E + 0] = Ex;
    E_xy[idx_E + 1] = Ey;
    E_xy[idx_E + 2] = Ez;
}

__global__ void kernel_xz(thrust::complex<float>* E_xz, thrust::complex<float>* W, float* z_layers, int* waves_begin, int* waves_end,
    float x_start, float z_start, float y, float d, int N, int layers) {
    int xi = blockDim.x * blockIdx.x + threadIdx.x;
    int zi = blockDim.y * blockIdx.y + threadIdx.y;
    if (xi >= N || zi >= N) return;

    unsigned int idx_E = zi * N * 3 + xi * 3;

    float x = x_start + d * xi;
    float z = z_start + d * zi;
    
    thrust::complex<float> Ex, Ey, Ez;
    
    evaluate(Ex, Ey, Ez, x, y, z, W, z_layers, waves_begin, waves_end, layers);

    E_xz[idx_E + 0] = Ex;
    E_xz[idx_E + 1] = Ey;
    E_xz[idx_E + 2] = Ez;
}

__global__ void kernel_yz(thrust::complex<float>* E_yz, thrust::complex<float>* W, float* z_layers, int* waves_begin, int* waves_end,
    float y_start, float z_start, float x, float d, int N, int layers) {
    int yi = blockDim.x * blockIdx.x + threadIdx.x;
    int zi = blockDim.y * blockIdx.y + threadIdx.y;
    if (yi >= N || zi >= N) return;

    unsigned int idx_E = zi * N * 3 + yi * 3;

    float y = y_start + d * yi;
    float z = z_start + d * zi;

    thrust::complex<float> Ex, Ey, Ez;

    evaluate(Ex, Ey, Ez, x, y, z, W, z_layers, waves_begin, waves_end, layers);

    E_yz[idx_E + 0] = Ex;
    E_yz[idx_E + 1] = Ey;
    E_yz[idx_E + 2] = Ez;
}

void gpu_cw_evaluate(thrust::complex<float>* E_xy, thrust::complex<float>* E_xz, thrust::complex<float>* E_yz,
    float x_start, float y_start, float z_start, float x, float y, float z, float d,
    int N, int device) {

    thrust::complex<float>* gpu_E;

    size_t E_size = N * N * sizeof(thrust::complex<float>) * 3;

    HANDLE_ERROR(cudaMemGetInfo(&free_gpu_memory, &total_gpu_memory));
    if(free_gpu_memory > E_size)
        HANDLE_ERROR(cudaMalloc(&gpu_E, E_size));
    else{
        std::cout<<"Insufficient GPU memory for an "<< N << "x" << N << "frame" <<std::endl;
        return;
    }

    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));
    int block_dim = sqrt(prop.maxThreadsPerBlock);
    dim3 threads(block_dim, block_dim);
    dim3 blocks(N / threads.x + 1, N / threads.y + 1);

    kernel_xy << <blocks, threads >> > (gpu_E, gpu_W, gpu_z_layers, gpu_waves_begin, gpu_waves_end, x_start, y_start, z, d, N, layers);
    HANDLE_ERROR(cudaMemcpy(E_xy, gpu_E, E_size, cudaMemcpyDeviceToHost));

    kernel_xz << <blocks, threads >> > (gpu_E, gpu_W, gpu_z_layers, gpu_waves_begin, gpu_waves_end, x_start, z_start, y, d, N, layers);
    HANDLE_ERROR(cudaMemcpy(E_xz, gpu_E, E_size, cudaMemcpyDeviceToHost));

    kernel_yz << <blocks, threads >> > (gpu_E, gpu_W, gpu_z_layers, gpu_waves_begin, gpu_waves_end, y_start, z_start, x, d, N, layers);
    HANDLE_ERROR(cudaMemcpy(E_yz, gpu_E, E_size, cudaMemcpyDeviceToHost));


    HANDLE_ERROR(cudaFree(gpu_E));

}
