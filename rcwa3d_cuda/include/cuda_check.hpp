#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// Wrap any CUDA, cuFFT, cuBLAS or cuSolver call: CHECK(cudaMalloc(...));
#define CHECK(call) CheckStatus((call), #call, __FILE__, __LINE__)

inline void Fail(const char* call, const char* file, int line, const char* message)
{
    /*
    Function that reports a failed call with its location and stops the program.
    */
    std::cerr << file << ':' << line << ": " << call << " failed: " << message << '\n';
    std::abort();
}

inline void CheckStatus(cudaError_t status, const char* call, const char* file, int line)
{
    /*
    Function that checks a CUDA runtime return code.
    */
    if (status != cudaSuccess) Fail(call, file, line, cudaGetErrorString(status));
}

inline void CheckStatus(cufftResult status, const char* call, const char* file, int line)
{
    /*
    Function that checks a cuFFT return code.
    */
    if (status != CUFFT_SUCCESS) Fail(call, file, line, ("cuFFT error " + std::to_string(status)).c_str());
}

inline void CheckStatus(cublasStatus_t status, const char* call, const char* file, int line)
{
    /*
    Function that checks a cuBLAS return code.
    */
    if (status != CUBLAS_STATUS_SUCCESS) Fail(call, file, line, cublasGetStatusString(status));
}

inline void CheckStatus(cusolverStatus_t status, const char* call, const char* file, int line)
{
    /*
    Function that checks a cuSolver return code.
    */
    if (status != CUSOLVER_STATUS_SUCCESS) Fail(call, file, line, ("cuSolver error " + std::to_string(status)).c_str());
}