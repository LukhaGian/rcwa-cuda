#include "rcwa_cuda.hpp"

void convMat_cu(cufftHandle& plan, cuda_Complex_d* d_er_ptr, cuda_Complex_d* d_ur_ptr, cuda_Complex_d* d_erc_ptr, cuda_Complex_d* d_urc_ptr, int Nx, int Ny, int num_layers, cudaStream_t stream)
{
    /*
    Function that 
    */
    // Target the specific stream instance for this cuFFT plan configuration
    cufftSetStream(plan, stream);

    int thread_per_block = 256;
    int shift_blocks = ((Nx / 2) * (Ny / 2) * num_layers + thread_per_block - 1) / thread_per_block;

    // potentially merge the calls into a unique FFT and QuadShift
    cufftExecZ2Z(plan, d_er_ptr, d_er_ptr, CUFFT_FORWARD);
    NormalizeQuadShiftFFT<<<shift_blocks, thread_per_block, 0, stream>>>(d_er_ptr, Nx, Ny, num_layers); // 256 threads, 32 threads per warp --> 8 warps per block
    cufftExecZ2Z(plan, d_ur_ptr, d_ur_ptr, CUFFT_FORWARD);
    NormalizeQuadShiftFFT<<<shift_blocks, thread_per_block, 0, stream>>>(d_ur_ptr, Nx, Ny, num_layers);

}


__global__ void NormalizeQuadShiftFFT(cuda_Complex_d* data, int Nx, int Ny, int num_layers)
{
    /*
    Kernel that defines the normalization for the FFT data and swaps opposite FFT quadrants for each layer.
    NOTE: kernel logic follows row major layout, as cuFFT's standard.
    Memory coalescing should be acheived by keeping warps within rows (if Nx/2 is a multiple of warp size)
    */

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

        int q1 = layer_offset + y * Nx + x; // first quadrant location in row major layout
        int q2 = layer_offset + y * Nx + x + half_Nx; // second quadrant location in row major layout
        int q3 = layer_offset + (y + half_Ny) * Nx + x; // third quadrant location in row major layout
        int q4 = layer_offset + (y + half_Ny) * Nx + x + half_Nx; // fourth quadrant location in row major layout

        // swap the quadrants and adjust scaling
        cuda_Complex_d a = data[q1];
        cuda_Complex_d b = data[q4];
        data[q1] = make_cuDoubleComplex(b.x * scale, b.y * scale);
        data[q4] = make_cuDoubleComplex(a.x * scale, a.y * scale);
        a = data[q2];
        b = data[q3];
        data[q2] = make_cuDoubleComplex(b.x * scale, b.y * scale);
        data[q3] = make_cuDoubleComplex(a.x * scale, a.y * scale);
    }
}