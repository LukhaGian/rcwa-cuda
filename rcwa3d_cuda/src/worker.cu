#include "worker.hpp"
#include <iostream>
#include <cstdlib>
#include <initializer_list> // passing a braced list of values to constructors and functions
#include <cstring> // std::memcpy

void InitWorker(StreamWorker& w, int gpu, const Device& device, const ValueTable& table, const RCWAParams& params, bool debug)
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

    
    // Check possible issues before uploading: a bad index would read random GPU memory without any error
    if (static_cast<int>(device.er_desc.size()) != num_layers || static_cast<int>(device.ur_desc.size()) != num_layers)
    {
        std::cerr << "InitWorker: need one er_desc and one ur_desc per layer" <<'\n';
        std::abort();
    }
    if (table.values.size() != static_cast<size_t>(table.num_values) * table.num_wavelengths)
    {
        std::cerr << "InitWorker: table size is not num_values * num_wavelengths" <<'\n';
        std::abort();
    }
    for (int l = 0; l < num_layers; ++l)
        for (const ShapeDesc* s : {&device.er_desc[l], &device.ur_desc[l]}) // using pointers
            if (s->type < SHAPE_UNIFORM || s->type > SHAPE_SQUARE ||
                s->value_in < 0 || s->value_in >= table.num_values ||
                s->value_out < 0 || s->value_out >= table.num_values)
            {
                std::cerr << "InitWorker: invalid ShapeDesc on layer " << l << '\n';
                std::abort();
            }

    // Upload descriptors and value table once (a few KB, pageable is ok)
    w.num_values = table.num_values; // no., of different materials used in the device
    w.num_wavelengths = table.num_wavelengths; // total no. of wavelengths
    const size_t desc_bytes = sizeof(ShapeDesc) * num_layers;
    const size_t values_bytes = sizeof(cuda_Complex_d) * table.values.size();
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_er_desc), desc_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_ur_desc), desc_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_values), values_bytes));
    CHECK(cudaMemcpyAsync(w.d_er_desc, device.er_desc.data(), desc_bytes, cudaMemcpyHostToDevice, w.stream));
    CHECK(cudaMemcpyAsync(w.d_ur_desc, device.ur_desc.data(), desc_bytes, cudaMemcpyHostToDevice, w.stream));
    CHECK(cudaMemcpyAsync(w.d_values, table.values.data(), values_bytes, cudaMemcpyHostToDevice, w.stream));
    CHECK(cudaStreamSynchronize(w.stream)); // the host vectors may change after InitWorker returns --> stream sync

}


void RunWorker(StreamWorker& w, const Device& device, const Source& source, const RCWAParams& params, std::atomic<int>& next_wavelength)
{
    /*
    Worker pulls wavelength indices from the shared counter until none are left and
    it solves each one of them  on this worker's stream.
    For now only the convolution matrices are computed.
    */

    CHECK(cudaSetDevice(w.gpu));
    const int num_wavelengths = static_cast<int>(source.lambda0.size());
    const int Q = 2 * params.Ny_harmonics + 1; // Note: y represents the rows of the matrices
    const int PQ = (2 * params.Nx_harmonics + 1) * Q;

    if (num_wavelengths != w.num_wavelengths)
    {
        std::cerr << "RunWorker: value table rows != number of wavelengths\n";
        std::abort();
    }
    // setting thread grid for BuildLayers
    // 1d flattening of thread blocks for each layer
    const dim3 grid((device.Nx * device.Ny + 255) / 256, device.num_layers); // x: pixels in a layer, y: layer

    while (true)
    {
        const int i = next_wavelength.fetch_add(1);
        if (i >= num_wavelengths)
            break;

        const cuda_Complex_d* values_i = w.d_values + static_cast<size_t>(i) * w.num_values; // row i, w.d_values points to first element
        // Build the layers
        BuildLayers<<<grid, 256, 0, w.stream>>>(w.d_er_desc, w.d_ur_desc, values_i, w.d_er, w.d_ur,
                                                device.Nx, device.Ny, device.Lx, device.Ly); // use 256 threads per block
        CHECK(cudaGetLastError());
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
        ++w.num_solved;
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
    CHECK(cudaFree(w.d_values));
    CHECK(cudaFree(w.d_ur_desc));
    CHECK(cudaFree(w.d_er_desc));
    CHECK(cudaFree(w.d_urc));
    CHECK(cudaFree(w.d_erc));
    CHECK(cudaFree(w.d_ur));
    CHECK(cudaFree(w.d_er));
    // cudaFreeHost since we are using pinned memory host side
    CHECK(cudaFreeHost(w.h_urc)); // nullptr is fine
    CHECK(cudaFreeHost(w.h_erc));
    // cuSolver
    CHECK(cusolverDnDestroy(w.solver));
    // cuBLAS
    CHECK(cublasDestroy(w.blas));
    // cuFFT plan
    CHECK(cufftDestroy(w.plan));
    CHECK(cudaStreamDestroy(w.stream)); // last to be destroyed: the handles were bound to it
    w = StreamWorker{}; // new default worker
}

