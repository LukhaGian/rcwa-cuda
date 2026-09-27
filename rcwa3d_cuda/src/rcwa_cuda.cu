#include "rcwa_cuda.hpp"

__global__ void normalizeFFT(cuda_Complex_d* data, int Nx, int Ny, int num_layers)
{
    /*
    Kernel that applies normalization to the FFT data.
    To be added as part of the FFT post-processing step as perhaps a callback function.
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = Nx * Ny * num_layers;
    if (idx < total_elements)
    {
        data[idx].x /= static_cast<cuda_Real_d>(Nx * Ny);
        data[idx].y /= static_cast<cuda_Real_d>(Nx * Ny);
    }
}