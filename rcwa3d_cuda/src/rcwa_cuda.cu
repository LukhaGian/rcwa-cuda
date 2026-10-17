#include "rcwa_cuda.hpp"

void convMat_cu(cufftHandle& plan, cuda_Complex_d* d_er_ptr, cuda_Complex_d* d_ur_ptr, int Nx, int Ny, int num_layers, cudaStream_t stream)
{
    // Target the specific stream instance for this cuFFT plan configuration
    cufftSetStream(plan, stream);

    int blocks = (Nx * Ny * num_layers + 255) / 256;
    int shift_blocks = ((Nx / 2) * (Ny / 2) * num_layers + 255) / 256;

    cufftExecZ2Z(plan, d_er_ptr, d_er_ptr, CUFFT_FORWARD);
    normalizeFFT<<<blocks, 256, 0, stream>>>(d_er_ptr, Nx, Ny, num_layers);
    fftQuadShift<<<shift_blocks, 256, 0, stream>>>(d_er_ptr, Nx, Ny, num_layers);
    cufftExecZ2Z(plan, d_ur_ptr, d_ur_ptr, CUFFT_FORWARD);
    normalizeFFT<<<blocks, 256, 0, stream>>>(d_ur_ptr, Nx, Ny, num_layers);
    fftQuadShift<<<shift_blocks, 256, 0, stream>>>(d_ur_ptr, Nx, Ny, num_layers);

}


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

// check if correct
__global__ void fftQuadShift(cuda_Complex_d* data, int Nx, int Ny, int num_layers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_Nx = Nx / 2;
    int half_Ny = Ny / 2;
    int elements_per_layer = half_Nx * half_Ny;
    int total_elements = elements_per_layer * num_layers;

    if (idx < total_elements)
    {
        int x = idx % half_Nx;
        int y = (idx / half_Nx) % half_Ny;
        int layer = idx / elements_per_layer;
        int layer_offset = layer * Nx * Ny;

        int q1 = layer_offset + y * Nx + x;
        int q2 = layer_offset + y * Nx + x + half_Nx;
        int q3 = layer_offset + (y + half_Ny) * Nx + x;
        int q4 = layer_offset + (y + half_Ny) * Nx + x + half_Nx;

        cuda_Complex_d value = data[q1];
        data[q1] = data[q4];
        data[q4] = value;

        value = data[q2];
        data[q2] = data[q3];
        data[q3] = value;
    }
}