#include "rcwa_cuda.hpp"

void convMat_cu(cufftHandle& plan, cuda_Complex_d* d_er_ptr, cuda_Complex_d* d_ur_ptr, cuda_Complex_d* d_erc_ptr, cuda_Complex_d* d_urc_ptr, int Nx, int Ny, int num_layers, int Nx_harmonics, int Ny_harmonics, int Q, int PQ, cudaStream_t stream)
{
    /*
    Function that computes the convolution matrices erc and urc (pointers d_erc_ptr and d_urc_ptr) 
    applying a forward FFT, normalization and quandrant shift to er and ur (pointers d_er_ptr and d_ur_ptr)
    and then computing the convolution matrices (Toeplitz).
    */

    // Target the specific stream instance for this cuFFT plan configuration
    cufftSetStream(plan, stream);

    int thread_per_block = 256;
    int shift_blocks = ((Nx / 2) * (Ny / 2) * num_layers + thread_per_block - 1) / thread_per_block;
    int conv_blocks = (num_layers * PQ * PQ + thread_per_block - 1) / thread_per_block; // check what can be more optimal

    // potentially merge the calls into a unique FFT and QuadShift
    cufftExecZ2Z(plan, d_er_ptr, d_er_ptr, CUFFT_FORWARD);
    NormalizeQuadShiftFFT<<<shift_blocks, thread_per_block, 0, stream>>>(d_er_ptr, Nx, Ny, num_layers); // 256 threads, 32 threads per warp --> 8 warps per block
    convMatAssembly<<<conv_blocks, thread_per_block, 0, stream>>>(d_er_ptr, d_erc_ptr, Nx, Ny, num_layers, Nx_harmonics, Ny_harmonics, Q, PQ);
    cufftExecZ2Z(plan, d_ur_ptr, d_ur_ptr, CUFFT_FORWARD);
    NormalizeQuadShiftFFT<<<shift_blocks, thread_per_block, 0, stream>>>(d_ur_ptr, Nx, Ny, num_layers);
    convMatAssembly<<<conv_blocks, thread_per_block, 0, stream>>>(d_ur_ptr, d_urc_ptr, Nx, Ny, num_layers, Nx_harmonics, Ny_harmonics, Q, PQ);

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

__global__ void convMatAssembly(cuda_Complex_d* fft_data, cuda_Complex_d* conv_data, int Nx, int Ny, int num_layers, int Nx_harmonics, int Ny_harmonics, int Q, int PQ)
{

    /*
    Kernel that assembles the convolution matrix from the fft data.
    */
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = PQ * PQ * num_layers;

    if (out_idx < total) 
    {
        int layer = out_idx / (PQ * PQ);
        int within_layer = out_idx % (PQ * PQ);

        // conv_data is row-major: n varies fastest
        int m = within_layer / PQ;
        int n = within_layer % PQ;

        int px_m = m / Q - Nx_harmonics;
        int py_m = m % Q - Ny_harmonics;
        int px_n = n / Q - Nx_harmonics;
        int py_n = n % Q - Ny_harmonics;

        int kx = (Nx / 2) + px_m - px_n;
        int ky = (Ny / 2) + py_m - py_n;

        int c_offset = out_idx;
        if (kx >= 0 && kx < Nx && ky >= 0 && ky < Ny) 
        {
            int f_offset = layer * Nx * Ny + ky * Nx + kx;
            conv_data[c_offset] = fft_data[f_offset];
        } 
        else // to be checked thoroughly
            conv_data[c_offset] = make_cuDoubleComplex(0.0, 0.0);
    }
}