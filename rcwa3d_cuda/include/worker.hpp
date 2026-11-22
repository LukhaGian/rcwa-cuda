#pragma once

#include "rcwa_cuda.hpp"
#include <atomic> // thread-safe atomic operations


// struct that defines one worker: a host thread drives one stream with its own handles and buffers on one GPU
struct StreamWorker
{
    int gpu{}; // worker's assigned GPU
    int num_solved{0}; // no. of wavelengths solved by the worker
    cudaStream_t stream{};
    cufftHandle plan{}; // bound to stream in InitWorker
    cublasHandle_t blas{}; // bound to stream in InitWorker
    cusolverDnHandle_t solver{}; // bound to stream in InitWorker
    // set sizes
    size_t fft_bytes{}; // Nx * Ny * num_layers cuda_Complex_d values
    size_t convMat_bytes{}; // PQ * PQ * num_layers cuda_Complex_d values
    // pinned host buffers
    cuda_Complex_d* h_erc{}; // testing only: nullptr unless debug flag
    cuda_Complex_d* h_urc{}; // testing only: nullptr unless debug flag
    bool has_debug{}; // testing only: h_erc/h_urc are filled
    // device buffers
    cuda_Complex_d* d_er{};
    cuda_Complex_d* d_ur{};
    cuda_Complex_d* d_erc{};
    cuda_Complex_d* d_urc{};
    // shape descriptor in device
    ShapeDesc* d_er_desc{};
    ShapeDesc* d_ur_desc{};
    cuda_Complex_d* d_values{}; // whole value table, row i = wavelength i
    int num_values{}; // no. columns of the table
    int num_wavelengths{}; // no. rows of the table
};

// Initialize the worker
void InitWorker(StreamWorker& w, int gpu, const Device& device, const ValueTable& table, const RCWAParams& params, bool debug);

// Run the worker
void RunWorker(StreamWorker& w, const Device& device, const Source& source, const RCWAParams& params, std::atomic<int>& next_wavelength);

// Destroy the worker
void DestroyWorker(StreamWorker& w);