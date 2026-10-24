#include "rcwa_cuda.hpp"

void convMat_cu(cufftHandle& plan, cuda_Complex_d* d_er_ptr, cuda_Complex_d* d_ur_ptr, int Nx, int Ny, int num_layers, cudaStream_t stream)
{
    // Target the specific stream instance for this cuFFT plan configuration
    cufftSetStream(plan, stream);

    int thread_per_block = 256;
    int blocks = (Nx * Ny * num_layers + thread_per_block - 1) / thread_per_block;
    int shift_blocks = ((Nx / 2) * (Ny / 2) * num_layers + thread_per_block - 1) / thread_per_block;

    cufftExecZ2Z(plan, d_er_ptr, d_er_ptr, CUFFT_FORWARD);
    normalizeFFT<<<blocks, thread_per_block, 0, stream>>>(d_er_ptr, Nx, Ny, num_layers);
    fftQuadShift<<<shift_blocks, thread_per_block, 0, stream>>>(d_er_ptr, Nx, Ny, num_layers); // 256 threads, 32 threads per warp --> 8 warps per block
    cufftExecZ2Z(plan, d_ur_ptr, d_ur_ptr, CUFFT_FORWARD);
    normalizeFFT<<<blocks, thread_per_block, 0, stream>>>(d_ur_ptr, Nx, Ny, num_layers);
    fftQuadShift<<<shift_blocks, thread_per_block, 0, stream>>>(d_ur_ptr, Nx, Ny, num_layers);

}


__global__ void normalizeFFT(cuda_Complex_d* data, int Nx, int Ny, int num_layers)
{
    /*
    Kernel that applies normalization to the FFT data.
    To be added as part of the FFT post-processing step as perhaps a callback function OR just add it to fftQuadShift kernel, it's obvious.
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
    // Be careful, it's row major as cuFFT standard
    // Memory coalescing should be acheived by keeping warps within rows (if Nx/2 is a multiple of warp size)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_Nx = Nx / 2;
    int half_Ny = Ny / 2;
    int elements_per_layer = half_Nx * half_Ny;
    int total_elements = elements_per_layer * num_layers;
    cuda_Real_d scale = 1.0 / static_cast<cuda_Real_d>(Nx * Ny); // Normalization factor for FFT

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
        /*
        double2 a = *reinterpret_cast<const double2*>(&data[q1]);
        double2 b = *reinterpret_cast<const double2*>(&data[q4]);
        *reinterpret_cast<double2*>(&data[q1]) = make_double2(b.x * scale, b.y * scale);
        *reinterpret_cast<double2*>(&data[q4]) = make_double2(a.x * scale, a.y * scale);
        cuda_Complex_d a = data[q1];
        cuda_Complex_d b = data[q4];
        data[q1] = make_cuDoubleComplex(b.x * scale, b.y * scale);
        data[q4] = make_cuDoubleComplex(a.x * scale, a.y * scale);
        */
    }
}