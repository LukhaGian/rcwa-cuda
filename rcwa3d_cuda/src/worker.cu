#include "worker.hpp"
#include <iostream>
#include <cstdlib>
#include <initializer_list> // passing a braced list of values to constructors and functions


WorkerSizes MakeSizes(const Device& device, const RCWAParams& params, const ValueTable& table)
{
    /*
    Function that returns the sizes of the problem for a worker to solve.
    */

    return WorkerSizes(device.Nx, device.Ny, device.num_layers, params.Nx_harmonics, params.Ny_harmonics,
                      table.num_values, table.num_wavelengths);
}


static void ProblemError(const char* message) // static: internal linkage, private to worker.cu
{
    /*
    Function that reports an invalid problem and stops the program.
    */

    std::cerr << "RunWorker: " << message << '\n';
    std::abort();
}


static void CheckProblem(const WorkerSizes& sz, const Device& device, const ValueTable& table,
                         const Source& source, const RCWAParams& params)
{
    /*
    Function that checks that the problem fits the worker's sizes and that every descriptor is valid.
    A bad index would read random GPU memory without any error, so check before uploading.
    */
    
    // device -- worker sizes Nx, Ny, num_layers
    if (device.Nx != sz.Nx || device.Ny != sz.Ny || device.num_layers != sz.num_layers)
        ProblemError("device grid or layer count differs from the worker's sizes");
    // params -- worker Nx_harmonics, Ny_harmonics
    if (params.Nx_harmonics != sz.Nx_harmonics || params.Ny_harmonics != sz.Ny_harmonics)
        ProblemError("harmonics differ from the worker's sizes");
    // device -- worker sizes length for er_desc and ur_desc (one description for each layer in [0, num_layers-1] for er and ur)
    if (static_cast<int>(device.er_desc.size()) != sz.num_layers || static_cast<int>(device.ur_desc.size()) != sz.num_layers)
        ProblemError("need one er_desc and one ur_desc per layer");
    // check table sizes
    if (table.values.size() != static_cast<size_t>(table.num_values) * table.num_wavelengths)
        ProblemError("table size is not num_values * num_wavelengths");
    // table -- worker sizes capacity
    if (table.num_values > sz.max_values || table.num_wavelengths > sz.max_wavelengths)
        ProblemError("value table exceeds the worker's capacity");
    // source -- table no. wavelengths
    if (static_cast<int>(source.lambda0.size()) != table.num_wavelengths)
        ProblemError("value table rows != number of wavelengths");
    // layer by layer check for shape descriptors for er and ur
    for (int l = 0; l < sz.num_layers; ++l)
        for (const ShapeDesc* s : {&device.er_desc[l], &device.ur_desc[l]})
            if (s->type < SHAPE_UNIFORM || s->type > SHAPE_SQUARE ||
                s->value_in < 0 || s->value_in >= table.num_values ||
                s->value_out < 0 || s->value_out >= table.num_values)
            {
                std::cerr << "RunWorker: invalid ShapeDesc on layer " << l << '\n';
                std::abort();
            }
}


void InitWorker(StreamWorker& w, int gpu, const WorkerSizes& sizes, bool debug)
{
    /* 
    Initialize the worker: binds the cuFFT plan, cuBLAS and cuSolver handles.
    Moreover, it allocates the pinned host and device buffers on GPU gpu.
    It depends only on the sizes in Workersizes.
    */

    // set the sizes
    w.sizes = sizes;
    // Set the GPU
    w.gpu = gpu;
    // Set device
    CHECK(cudaSetDevice(w.gpu));
    // Create a non-blocking stream wrt legacy default stream
    CHECK(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    // Set RCWA's dimensions
    const int Nx = sizes.Nx;
    const int Ny = sizes.Ny;
    const int num_layers = sizes.num_layers;
    const int PQ = (2 * sizes.Nx_harmonics + 1) * (2 * sizes.Ny_harmonics + 1);
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
    // Device work buffers
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_er), w.fft_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_ur), w.fft_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_erc), w.convMat_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_urc), w.convMat_bytes));

    
    // Problem buffers, by capacity: filled in RunWorker
    const size_t desc_bytes = sizeof(ShapeDesc) * num_layers;
    const size_t values_bytes = sizeof(cuda_Complex_d) * static_cast<size_t>(sizes.max_values) * sizes.max_wavelengths;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_er_desc), desc_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_ur_desc), desc_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&w.d_values), values_bytes));
}


void RunWorker(StreamWorker& w, const Device& device, const ValueTable& table,
               const Source& source, const RCWAParams& params, std::atomic<int>& next_wavelength, int debug_target)
{
    /*
    Worker checks and uploads the problem (descriptors and value table), then pulls wavelength
    indices from the shared counter until none are left and solves each one on this worker's stream.
    For now only the convolution matrices are computed.
    */

    CHECK(cudaSetDevice(w.gpu));
    CheckProblem(w.sizes, device, table, source, params); // check before loading that sizes match between the worker and current problem
    // Load the problem: a few KB, pageable is ok
    const size_t desc_bytes = sizeof(ShapeDesc) * w.sizes.num_layers;
    const size_t values_bytes = sizeof(Complex) * table.values.size(); // upload the current table, not the max capacity
    CHECK(cudaMemcpyAsync(w.d_er_desc, device.er_desc.data(), desc_bytes, cudaMemcpyHostToDevice, w.stream));
    CHECK(cudaMemcpyAsync(w.d_ur_desc, device.ur_desc.data(), desc_bytes, cudaMemcpyHostToDevice, w.stream));
    CHECK(cudaMemcpyAsync(w.d_values, table.values.data(), values_bytes, cudaMemcpyHostToDevice, w.stream));
    CHECK(cudaStreamSynchronize(w.stream)); // the caller may change the host vectors after the run --> sync
    
    w.num_values = table.num_values; // get the number of values of the loaded table, potentially smaller than the preallocated w.sizes.max_values
    w.num_wavelengths = table.num_wavelengths;
    w.num_solved = 0;
    w.debug_wavelength = -1; // debug output always belongs to the last run
    const int Nx = w.sizes.Nx;
    const int Ny = w.sizes.Ny;
    const int Q = 2 * w.sizes.Ny_harmonics + 1; // Note: y represents the rows of the matrices
    const int PQ = (2 * w.sizes.Nx_harmonics + 1) * Q;
    // 1d flattening of thread blocks for each layer
    const dim3 grid((Nx * Ny + 255) / 256, w.sizes.num_layers); // x: pixels in a layer, y: layer

    while (true)
    {
        const int i = next_wavelength.fetch_add(1);
        if (i >= w.num_wavelengths)
            break;

        const cuda_Complex_d* values_i = w.d_values + static_cast<size_t>(i) * w.num_values; // row i, w.d_values points to first element
        // Build the layers
        BuildLayers<<<grid, 256, 0, w.stream>>>(w.d_er_desc, w.d_ur_desc, values_i, w.d_er, w.d_ur,
                                                Nx, Ny, device.Lx, device.Ly); // use 256 threads per block
        CHECK(cudaGetLastError());
        convMat_cu(w.plan, w.d_er, w.d_ur, w.d_erc, w.d_urc, Nx, Ny, 
                   w.sizes.num_layers, w.sizes.Nx_harmonics, w.sizes.Ny_harmonics, Q, PQ, w.stream);

        // WIP: wave vectors, gap medium, layer S-matrices, R/T for wavelength i

        if (w.h_erc != nullptr && i == debug_target)
        {
            CHECK(cudaMemcpyAsync(w.h_erc, w.d_erc, w.convMat_bytes, cudaMemcpyDeviceToHost, w.stream));
            CHECK(cudaMemcpyAsync(w.h_urc, w.d_urc, w.convMat_bytes, cudaMemcpyDeviceToHost, w.stream));
            w.debug_wavelength = i;
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

