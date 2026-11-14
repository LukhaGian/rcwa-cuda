#include "rcwa_cuda.hpp"
#include<cstring> // std::memcpy


void InitWorker(StreamWorker& w, int gpu, const Device& device, const RCWAParams& params, bool debug)
{
    /* 
    Initialize the worker: binds the cuFFT plan, cuBLAS and cuSolver handles.
    Moreover, it allocates the pinned host and device buffers on GPU gpu.
    */

    // Set the GPU
    w.gpu = gpu;
    // Set device
    CHECK(cudaSetDevice(w.gpu));
    // Create a non-blocking stream wrt legacy default stream
    CHECK(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    // Set RCWA's dimensions
    const int Nx = device.Nx;
    const int Ny = device.Ny;
    const int num_layers = device.num_layers;
    const int PQ = (2 * params.Nx_harmonics + 1) * (2 * params.Ny_harmonics + 1);
    // Set byte sizes
    w.fft_bytes = sizeof(cuda_Complex_d) * static_cast<size_t>(Nx) * Ny * num_layers;
    w.convMat_bytes = sizeof(cuda_Complex_d) * static_cast<size_t>(PQ) * PQ * num_layers;
    
    // Batched 2D FFT over all layers, plan bound to worker w's stream
    int rank = 2; // 2D FFT
    int* inembed = NULL;
    int* onembed = NULL;
    int dims[2] = {Ny, Nx};
    int istride = 1;
    int idist = Nx*Ny;
    int ostride = 1;
    int odist = Nx*Ny;
    // Create a cuFFT plan --> many for batched FFT
    CHECK(cufftPlanMany(&w.plan, rank, dims, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, num_layers));
    CHECK(cufftSetStream(w.plan, w.stream));
    // cuBLAS
    CHECK(cublasCreate(&w.blas));
    CHECK(cublasSetStream(w.blas, w.stream));
    // cuSolver
    CHECK(cusolverDnCreate(&w.solver));
    CHECK(cusolverDnSetStream(w.solver, w.stream));

    // Pinned host buffers
    // for now er/ur don't depend on the wavelength, so fill them just once
    CHECK(cudaMallocHost(reinterpret_cast<void**>(&w.h_er), w.fft_bytes));
    CHECK(cudaMallocHost(reinterpret_cast<void**>(&w.h_ur), w.fft_bytes));
    std::memcpy(w.h_er, device.er.data(), w.fft_bytes);
    std::memcpy(w.h_ur, device.ur.data(), w.fft_bytes);
    if (debug)
    {
        CHECK(cudaMallocHost(reinterpret_cast<void**>(&w.h_erc), w.convMat_bytes));
        CHECK(cudaMallocHost(reinterpret_cast<void**>(&w.h_urc), w.convMat_bytes));
    }
    // Device buffers
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_er), w.fft_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_ur), w.fft_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_erc), w.convMat_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_urc), w.convMat_bytes));
}


void RunWorker(StreamWorker& w, const Device& device, const Source& source, const RCWAParams& params, std::atomic<int>& next_wavelength)
{
    /*
    Worker pulls wavelenth indices from the shared counter until none are left and
    it solves each one of them  on this worker's stream.
    For now only the convolution matrices are computed.
    */

    CHECK(cudaSetDevice(w.gpu));
    const int num_wavelengths = static_cast<int>(source.lambda0.size());
    const int Q = 2 * params.Ny_harmonics + 1; // Note: y represents the rows of the matrices
    const int PQ = (2 * params.Nx_harmonics + 1) * Q;

    while (true)
    {
        const int i = next_wavelength.fetch_add(1);
        if (i >= num_wavelengths)
            break;

        // WIP: fill h_er/h_ur for lambda0[i] here: safe because the previous wavelength was synchronized
        CHECK(cudaMemcpyAsync(w.d_er, w.h_er, w.fft_bytes, cudaMemcpyHostToDevice, w.stream));
        CHECK(cudaMemcpyAsync(w.d_ur, w.h_ur, w.fft_bytes, cudaMemcpyHostToDevice, w.stream));
        convMat_cu(w.plan, w.d_er, w.d_ur, w.d_erc, w.d_urc, device.Nx, device.Ny, 
                   device.num_layers, params.Nx_harmonics, params.Ny_harmonics, Q, PQ, w.stream);

        // WIP: wave vectors, gap medium, layer S-matrices, R/T for wavelength i

        if (w.h_erc != nullptr && !w.has_debug)
        {
            CHECK(cudaMemcpyAsync(w.h_erc, w.d_erc, w.convMat_bytes, cudaMemcpyDeviceToHost, w.stream));
            CHECK(cudaMemcpyAsync(w.h_urc, w.d_urc, w.convMat_bytes, cudaMemcpyDeviceToHost, w.stream));
            w.has_debug = true;
        }
        CHECK(cudaStreamSynchronize(w.stream)); // wavelength i is complete
    }
}


void DestroyWorker(StreamWorker& w)
{
    /*
    Function that releases everything InitWorker created, in reverse order
    and it resets the struct.
    */

    CHECK(cudaSetDevice(w.gpu));
    CHECK(cudaStreamSynchronize(w.stream)); // nothing may still be using the buffers
    CHECK(cudaFree(w.d_urc));
    CHECK(cudaFree(w.d_erc));
    CHECK(cudaFree(w.d_ur));
    CHECK(cudaFree(w.d_er));
    // cudaFreeHost since we are using pinned memory host side
    CHECK(cudaFreeHost(w.h_urc)); // nullptr is fine
    CHECK(cudaFreeHost(w.h_erc));
    CHECK(cudaFreeHost(w.h_ur));
    CHECK(cudaFreeHost(w.h_er));
    // cuSolver
    CHECK(cusolverDnDestroy(w.solver));
    // cuBLAS
    CHECK(cublasDestroy(w.blas));
    // cuFFT plan
    CHECK(cufftDestroy(w.plan));
    CHECK(cudaStreamDestroy(w.stream)); // last to be destroyed: the handles were bound to it
    w = StreamWorker{}; // new default worker
}


void convMat_cu(cufftHandle& plan, cuda_Complex_d* d_er_ptr, cuda_Complex_d* d_ur_ptr, cuda_Complex_d* d_erc_ptr, cuda_Complex_d* d_urc_ptr, int Nx, int Ny, int num_layers, int Nx_harmonics, int Ny_harmonics, int Q, int PQ, cudaStream_t stream)
{
    /*
    Function that computes the convolution matrices erc and urc (pointers d_erc_ptr and d_urc_ptr) 
    applying a forward FFT, normalization and quandrant shift to er and ur (pointers d_er_ptr and d_ur_ptr)
    and then computing the convolution matrices (Toeplitz).
    Note: The plan must already be bound to stream (cufftSetStream in InitWorker).
    */


    int thread_per_block = 256;
    int shift_blocks = ((Nx / 2) * (Ny / 2) * num_layers + thread_per_block - 1) / thread_per_block;
    int conv_blocks = (num_layers * PQ * PQ + thread_per_block - 1) / thread_per_block; // check what can be more optimal

    CHECK(cufftExecZ2Z(plan, d_er_ptr, d_er_ptr, CUFFT_FORWARD));
    NormalizeQuadShiftFFT<<<shift_blocks, thread_per_block, 0, stream>>>(d_er_ptr, Nx, Ny, num_layers); // 256 threads, 32 threads per warp --> 8 warps per block
    CHECK(cudaGetLastError());
    convMatAssembly<<<conv_blocks, thread_per_block, 0, stream>>>(d_er_ptr, d_erc_ptr, Nx, Ny, num_layers, Nx_harmonics, Ny_harmonics, Q, PQ);
    CHECK(cudaGetLastError());
    CHECK(cufftExecZ2Z(plan, d_ur_ptr, d_ur_ptr, CUFFT_FORWARD));
    NormalizeQuadShiftFFT<<<shift_blocks, thread_per_block, 0, stream>>>(d_ur_ptr, Nx, Ny, num_layers);
    CHECK(cudaGetLastError());
    convMatAssembly<<<conv_blocks, thread_per_block, 0, stream>>>(d_ur_ptr, d_urc_ptr, Nx, Ny, num_layers, Nx_harmonics, Ny_harmonics, Q, PQ);
    CHECK(cudaGetLastError());

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

        // conv_data is column-major: m varies fastest --> cast to column-major indexing for cuBLAS/cuSOLVER calls
        int n = within_layer / PQ;
        int m = within_layer % PQ;

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
        else
            conv_data[c_offset] = make_cuDoubleComplex(0.0, 0.0);
    }
}