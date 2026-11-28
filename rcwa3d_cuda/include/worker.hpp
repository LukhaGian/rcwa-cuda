#pragma once

#include "rcwa_cuda.hpp"
#include <atomic> // thread-safe atomic operations

// struct that defines what a worker is bound to: dimensions and capacities, no geometry or materials
struct WorkerSizes
{
    int Nx{};
    int Ny{};
    int num_layers{};
    int Nx_harmonics{};
    int Ny_harmonics{};
    int max_values{}; // value maximum table capacity: columns -> max allocation
    int max_wavelengths{}; // value maximum table capacity: rows -> max allocation

    WorkerSizes()
    {
    }
    WorkerSizes(int Nx_, int Ny_, int num_layers_, int Nx_harmonics_, 
            int Ny_harmonics_, int max_values_, int max_wavelengths_)
            : Nx(Nx_), Ny(Ny_), num_layers(num_layers_), Nx_harmonics(Nx_harmonics_),
                Ny_harmonics(Ny_harmonics_), max_values(max_values_), max_wavelengths(max_wavelengths_)
    {
    }
};

// struct that defines one worker: a host thread drives one stream with its own handles and buffers on one GPU
struct StreamWorker
{
    WorkerSizes sizes{}; // set in InitWorker, fixed for the worker's lifetime
    int gpu{}; // worker's assigned GPU
    int num_solved{0}; // no. of wavelengths solved by the worker
    cudaStream_t stream{};
    cufftHandle plan{}; // bound to stream in InitWorker
    cublasHandle_t blas{}; // bound to stream in InitWorker
    cusolverDnHandle_t solver{}; // bound to stream in InitWorker
    // byte sizes
    size_t fft_bytes{}; // Nx * Ny * num_layers cuda_Complex_d values
    size_t convMat_bytes{}; // PQ * PQ * num_layers cuda_Complex_d values
    // pinned host buffers
    cuda_Complex_d* h_erc{}; // testing only: nullptr unless debug flag
    cuda_Complex_d* h_urc{}; // testing only: nullptr unless debug flag
    bool has_debug{}; // testing only: h_erc/h_urc are filled
    // device work buffers
    cuda_Complex_d* d_er{};
    cuda_Complex_d* d_ur{};
    cuda_Complex_d* d_erc{};
    cuda_Complex_d* d_urc{};
    // shape descriptors in device, allocated by capacity in InitWorker, filled by RunWorker
    ShapeDesc* d_er_desc{};
    ShapeDesc* d_ur_desc{};
    cuda_Complex_d* d_values{}; // whole value table, row i = wavelength i
    int num_values{}; // no. columns of the loaded table (current when calling RunWorker, <= sizes.max_values), row stride of d_val
    int num_wavelengths{}; // no. rows of the loaded table (current when calling RunWorker, <= sizes.max_wavelengths))
};

// Sizes of a problem, to initialize workers that can solve it
WorkerSizes MakeSizes(const Device& device, const RCWAParams& params, const ValueTable& table);

// Initialize the worker -> it depends only on the sizes
void InitWorker(StreamWorker& w, int gpu, const WorkerSizes& sizes, bool debug);

// Run the worker: load the problem into the worker, then solve wavelengths from the shared counter
void RunWorker(StreamWorker& w, const Device& device, const ValueTable& table, const Source& source, const RCWAParams& params, std::atomic<int>& next_wavelength);

// Destroy the worker
void DestroyWorker(StreamWorker& w);