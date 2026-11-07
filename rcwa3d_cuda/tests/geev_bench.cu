#include "cuda_check.hpp"
#include <chrono>
#include <complex>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

// Standalone benchmark of cusolverDnXgeev on random complex matrices.
// Usage: ./geev_bench [n1 n2 ...]   (default sizes: 500 1000 2000 3000 5202)
// Output: one CSV line per run; run 0 is a warm-up and should be ignored.

int main(int argc, char** argv)
{
    /*
    Times the eigensolve (eigenvalues and right eigenvectors) for each size,
    reports CPU cores used, workspace sizes, info, and the relative residual
    ||A V - V diag(W)||_F / (||A||_F ||V||_F) computed with cuBLAS.
    */
    std::vector<int64_t> sizes{500, 1000, 2000, 3000, 5202};
    if (argc > 1)
    {
        sizes.clear();
        for (int a = 1; a < argc; ++a) sizes.push_back(std::atoll(argv[a]));
    }
    const int reps = 3; // timed runs per size, after one warm-up

    cudaStream_t stream;
    CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)); // waiting threads sleep instead of spinning
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cusolverDnHandle_t solver;
    CHECK(cusolverDnCreate(&solver));
    CHECK(cusolverDnSetStream(solver, stream));
    cusolverDnParams_t solver_params;
    CHECK(cusolverDnCreateParams(&solver_params));
    cublasHandle_t blas;
    CHECK(cublasCreate(&blas));
    CHECK(cublasSetStream(blas, stream));

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    const cuDoubleComplex minus_one = make_cuDoubleComplex(-1.0, 0.0);

    std::cout << "n,run,wall_s,cores_used,device_ws_MB,host_ws_MB,info,residual\n";

    for (const int64_t n : sizes)
    {
        const size_t elements = static_cast<size_t>(n) * n;
        const size_t bytes = elements * sizeof(cuDoubleComplex);

        // Random complex matrix, entries uniform in [-1, 1] for both parts
        std::vector<std::complex<double>> h_A(elements);
        for (auto& a : h_A) a = std::complex<double>(dist(rng), dist(rng));

        cuDoubleComplex* d_A0 = nullptr;  // untouched copy of A, for the residual
        cuDoubleComplex* d_A = nullptr;   // overwritten by geev
        cuDoubleComplex* d_W = nullptr;   // eigenvalues
        cuDoubleComplex* d_VR = nullptr;  // right eigenvectors, column-major
        cuDoubleComplex* d_tmp = nullptr; // residual workspace
        int* d_info = nullptr;
        CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A0), bytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&d_W), sizeof(cuDoubleComplex) * n));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&d_VR), bytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tmp), bytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));
        CHECK(cudaMemcpyAsync(d_A0, h_A.data(), bytes, cudaMemcpyHostToDevice, stream));

        size_t ws_device = 0;
        size_t ws_host = 0;
        CHECK(cusolverDnXgeev_bufferSize(solver, solver_params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR, n,
                                         CUDA_C_64F, d_A, n, CUDA_C_64F, d_W, CUDA_C_64F, nullptr, n,
                                         CUDA_C_64F, d_VR, n, CUDA_C_64F, &ws_device, &ws_host));
        void* d_ws = nullptr;
        void* h_ws = nullptr;
        CHECK(cudaMalloc(&d_ws, ws_device > 0 ? ws_device : 1));
        CHECK(cudaMallocHost(&h_ws, ws_host > 0 ? ws_host : 1)); // pinned, as recommended for the hybrid algorithm

        for (int run = 0; run <= reps; ++run)
        {
            CHECK(cudaMemcpyAsync(d_A, d_A0, bytes, cudaMemcpyDeviceToDevice, stream));
            CHECK(cudaStreamSynchronize(stream)); // the copy is not part of the timing

            const auto wall_start = std::chrono::steady_clock::now();
            const std::clock_t cpu_start = std::clock(); // CPU time summed over all threads of the process

            CHECK(cusolverDnXgeev(solver, solver_params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR, n,
                                  CUDA_C_64F, d_A, n, CUDA_C_64F, d_W, CUDA_C_64F, nullptr, n,
                                  CUDA_C_64F, d_VR, n, CUDA_C_64F,
                                  d_ws, ws_device, h_ws, ws_host, d_info));
            CHECK(cudaStreamSynchronize(stream));

            const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
            const double cpu_s = static_cast<double>(std::clock() - cpu_start) / CLOCKS_PER_SEC;

            int h_info = 0;
            CHECK(cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));

            double residual = -1.0; // only computed on the last run
            if (run == reps && h_info == 0)
            {
                const int ni = static_cast<int>(n); // cuBLAS takes int dimensions
                const int64_t count = static_cast<int64_t>(elements);
                // tmp = A0 * VR
                CHECK(cublasZgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ni, ni, &one, d_A0, ni, d_VR, ni, &zero, d_tmp, ni));
                // A = VR * diag(W), reusing the buffer geev overwrote
                CHECK(cublasZdgmm(blas, CUBLAS_SIDE_RIGHT, ni, ni, d_VR, ni, d_W, 1, d_A, ni));
                // tmp = tmp - A
                CHECK(cublasZaxpy(blas, static_cast<int>(count), &minus_one, d_A, 1, d_tmp, 1));
                double norm_res = 0.0;
                double norm_A = 0.0;
                double norm_V = 0.0;
                CHECK(cublasDznrm2(blas, static_cast<int>(count), d_tmp, 1, &norm_res)); // host pointer mode: waits for the result
                CHECK(cublasDznrm2(blas, static_cast<int>(count), d_A0, 1, &norm_A));
                CHECK(cublasDznrm2(blas, static_cast<int>(count), d_VR, 1, &norm_V));
                residual = norm_res / (norm_A * norm_V);
            }

            std::cout << n << ',' << run << ',' << wall_s << ',' << cpu_s / wall_s << ','
                      << ws_device / 1e6 << ',' << ws_host / 1e6 << ',' << h_info << ',' << residual << std::endl;
        }

        CHECK(cudaFreeHost(h_ws));
        CHECK(cudaFree(d_ws));
        CHECK(cudaFree(d_info));
        CHECK(cudaFree(d_tmp));
        CHECK(cudaFree(d_VR));
        CHECK(cudaFree(d_W));
        CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_A0));
    }

    CHECK(cublasDestroy(blas));
    CHECK(cusolverDnDestroyParams(solver_params));
    CHECK(cusolverDnDestroy(solver));
    CHECK(cudaStreamDestroy(stream));
    return 0;
}