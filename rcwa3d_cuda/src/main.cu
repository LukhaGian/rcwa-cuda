#include "rcwa_cuda.hpp"
#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        printf("Hello from CUDA Kernel on thread %d!\n", tid);
    }
}

int main()
{
    // spatial grid for the layers
    int Nx = 5; // make sure it's always even
    int Ny = 5;

    int Nx_harmonics = 2;
    int Ny_harmonics = 2;

    Real Lx = 1.0; // unit cell dimension along X axis
    Real Ly = 1.0; // unit cell dimension along Y axis

    // Build the device layers
    std::vector<std::vector<Complex>> er_layers = 
    {
            /*
            layer::triangle_pattern(Nx, Ny, Lx, Ly),
            layer::x_pattern(Nx, Ny, Lx, Ly),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::square_pattern(Nx, Ny, Complex(2.0, 0.0), Complex(5.0, 0.0)),
            layer::y_pattern(Nx, Ny, Lx, Ly),
            layer::uniform(Nx, Ny, Complex(4.0, 0.0)),
            layer::square_pattern(Nx, Ny, Complex(7.0, 0.0), Complex(1.0, 0.0))
            */
            layer::triangle_pattern(Nx, Ny, Lx, Ly),
            layer::x_pattern(Nx, Ny, Lx, Ly),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0))


    };
    std::vector<std::vector<Complex>> ur_layers = 
    {
            /*
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0))
            */
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0))


    };
    std::vector<Complex> er = layer::stack(er_layers);
    std::vector<Complex> ur = layer::stack(ur_layers);

    //std::vector<Real> thickness{1.5, 0.3, 2.4, 2.6, 1.1, 0.25};
    std::vector<Real> thickness{1.5, 0.3, 2.4};

    // Build the device
    Device device(Nx, Ny, thickness.size(), Lx, Ly, er, ur, thickness, Nx_harmonics, Ny_harmonics);

    // Build all the sources that we will simulate
    size_t length = 100;  // # of sources to simulate

    std::vector<Real> lambda0_v(length);
    std::vector<Real> k0_v(length);
    Real lambda_start = 1.0;
    Real lambda_end = 2.0;
    // If length is 1, just set it to the lambda_start value to avoid division by zero
    if (length == 1)
    {
        lambda0_v[0] = lambda_start;
        k0_v[0] = 2 * M_PI / lambda0_v[0];
    }
    else 
    {
        Real step = (lambda_end - lambda_start) / (length - 1);
        for (size_t i = 0; i < length; ++i) 
        {
            lambda0_v[i] = lambda_start + i * step;
            k0_v[i] = 2 * M_PI / lambda0_v[i];
        }
    }

    Real theta = M_PI / 6.0;
    std::vector<Real> theta_v(length, theta); // All thetas are the same for now
    Real phi = M_PI / 12.0;
    std::vector<Real> phi_v(length, phi); // All phis are the same for now
    Real pte = 0.0; // TE pol
    std::vector<Real> pte_v(length, pte); // All ptes are the same for now
    Real ptm = 1.0; // TM pol
    std::vector<Real> ptm_v(length, ptm); // All ptms are the same for now
    Source source(lambda0_v, k0_v, theta_v, phi_v, pte_v, ptm_v);


    // Build reflection and transmission regions --> for now the materials are non wavelength dependent, extension to wavelength dependent materials is possible
    Complex er_ref(2.0, 0.0);
    Complex ur_ref(1.0, 0.0);
    Complex er_trn(9.0, 0.0);
    Complex ur_trn(1.0, 0.0);
    RCWAParams params(Nx_harmonics, Ny_harmonics, er_ref, ur_ref, er_trn, ur_trn);

    // CUDA

    // Safe pointer cast
    cuda_Complex_d* h_er_ptr = reinterpret_cast<cuda_Complex_d*>(er.data()); // layout compatibile types, .data() since we want to point to the underlying data, not the wrapper 
    std::cout << "Pointer to er array: " << static_cast<void*>(h_er_ptr) << std::endl;
    cuda_Complex_d* h_ur_ptr = reinterpret_cast<cuda_Complex_d*>(ur.data()); // layout compatibile types
    std::cout << "Pointer to ur array: " << static_cast<void*>(h_ur_ptr) << std::endl;

    cuda_Complex_d* d_er_ptr = nullptr;
    cuda_Complex_d* d_ur_ptr = nullptr;
    
    // Allocate memory on the GPU for er and ur arrays
    size_t total_bytes = sizeof(cuda_Complex_d) * Nx * Ny * thickness.size();
    cudaMalloc(reinterpret_cast<void**>(&d_er_ptr), total_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_ur_ptr), total_bytes);
    // Copy data from host to device
    cudaMemcpy(d_er_ptr, h_er_ptr, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ur_ptr, h_ur_ptr, total_bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;

    int rank = 2; //2D FFT
    int* inembed = NULL;
    int* onembed = NULL;
    int rank_dim[2] = {Ny, Nx};
    int istride = 1;
    int idist = Nx*Ny;
    int ostride = 1;
    int odist = Nx*Ny;
    cufftPlanMany(&plan, rank, rank_dim, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, thickness.size());

    cufftExecZ2Z(plan, d_er_ptr, d_er_ptr, CUFFT_FORWARD); // Launched all on default stream, also second cuffExec--> merge them
    normalizeFFT<<<(Nx*Ny*thickness.size() + 255)/256, 256>>>(d_er_ptr, Nx, Ny, thickness.size());
    cufftExecZ2Z(plan, d_ur_ptr, d_ur_ptr, CUFFT_FORWARD);

    cudaDeviceSynchronize();

    cudaMemcpy(h_er_ptr, d_er_ptr, total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ur_ptr, d_ur_ptr, total_bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_er_ptr);
    cudaFree(d_ur_ptr);

    std::cout << "Size of er " << er.size() << '\n'; /////////////////// To Be checked
    for (size_t i = 0; i < Nx*Ny*thickness.size(); ++i) 
    {
        if (i % (Nx*Ny) == 0)
            std::cout << "Reached end of layer" << '\n';
        std::cout << er[i] << '\n';
    }

    /*
    // Test
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running on GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    // Launch kernel with 1 block of 64 threads
    helloFromGPU<<<1, 64>>>();
    
    // Wait for GPU to complete
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    */

    return 0;
}