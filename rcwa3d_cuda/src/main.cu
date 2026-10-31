#include "rcwa_cuda.hpp"
#include <iostream>
#include <cuda_runtime.h>


int main()
{
    // spatial grid for the layers
    int Nx = 6; // make sure it's always even
    int Ny = 6;

    int Nx_harmonics = 1;
    int Ny_harmonics = 1;

    int P = 2 * Nx_harmonics + 1; // total number of harmonics along x axis
    int Q = 2 * Ny_harmonics + 1; // total number of harmonics along y axis
    int PQ = P * Q;

    Real Lx = 1.0; // unit cell dimension along x axis
    Real Ly = 1.0; // unit cell dimension along y axis

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
    // sizeof() returns a size_t, so no need for explicit cast
    const size_t total_bytes_FFT = sizeof(cuda_Complex_d) * Nx * Ny * thickness.size();
    const size_t convMat_elements = static_cast<size_t>(PQ) * PQ * thickness.size();
    const size_t total_bytes_convMat = sizeof(cuda_Complex_d) * convMat_elements;

    // Host pointers
    // Safe pointer cast
    cuda_Complex_d* h_er_ptr = reinterpret_cast<cuda_Complex_d*>(er.data()); // layout compatibile types, .data() since we want to point to the underlying data, not the wrapper 
    //std::cout << "Pointer to er array: " << static_cast<void*>(h_er_ptr) << std::endl;
    cuda_Complex_d* h_ur_ptr = reinterpret_cast<cuda_Complex_d*>(ur.data()); // layout compatibile types
    //std::cout << "Pointer to ur array: " << static_cast<void*>(h_ur_ptr) << std::endl;
    std::vector<Complex> h_erc(convMat_elements);
    std::vector<Complex> h_urc(convMat_elements);
    cuda_Complex_d* h_erc_ptr = reinterpret_cast<cuda_Complex_d*>(h_erc.data()); // allocate host side memory
    cuda_Complex_d* h_urc_ptr = reinterpret_cast<cuda_Complex_d*>(h_urc.data());

    // Device pointers
    cuda_Complex_d* d_er_ptr = nullptr;
    cuda_Complex_d* d_ur_ptr = nullptr;
    cuda_Complex_d* d_erc_ptr = nullptr;
    cuda_Complex_d* d_urc_ptr = nullptr;
    
    // Allocate memory on the GPU for er and ur matrices (will become FFTs)
    cudaMalloc(reinterpret_cast<void**>(&d_er_ptr), total_bytes_FFT);
    cudaMalloc(reinterpret_cast<void**>(&d_ur_ptr), total_bytes_FFT);
    // Copy data from host to device
    cudaMemcpy(d_er_ptr, h_er_ptr, total_bytes_FFT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ur_ptr, h_ur_ptr, total_bytes_FFT, cudaMemcpyHostToDevice);
    // Allocate memory on the GPU for convolution matrices erc and urc 
    cudaMalloc(reinterpret_cast<void**>(&d_erc_ptr), total_bytes_convMat);
    cudaMalloc(reinterpret_cast<void**>(&d_urc_ptr), total_bytes_convMat);

    // cuFFT plan
    cufftHandle plan;

    int rank = 2; //2D FFT
    int* inembed = NULL;
    int* onembed = NULL;
    int rank_dim[2] = {Ny, Nx};
    int istride = 1;
    int idist = Nx*Ny;
    int ostride = 1;
    int odist = Nx*Ny;

    // Create a cuFFT plan 
    cufftPlanMany(&plan, rank, rank_dim, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, thickness.size());

    // Perform convolution
    convMat_cu(plan, d_er_ptr, d_ur_ptr, d_erc_ptr, d_urc_ptr, Nx, Ny, thickness.size(), Nx_harmonics, Ny_harmonics, Q, PQ);

    cudaDeviceSynchronize();

    cudaMemcpy(h_er_ptr, d_er_ptr, total_bytes_FFT, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ur_ptr, d_ur_ptr, total_bytes_FFT, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_erc_ptr, d_erc_ptr, total_bytes_convMat, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_urc_ptr, d_urc_ptr, total_bytes_convMat, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_er_ptr);
    cudaFree(d_ur_ptr);
    cudaFree(d_erc_ptr);
    cudaFree(d_urc_ptr);

    //////////////////////////////////////////////////
    // TESTING
    //////////////////////////////////////////////////


    // fft
    /*
    std::cout << "Size of er " << er.size() << '\n'; /////////////////// To Be checked

    for (size_t i = 0; i < Nx*Ny*thickness.size(); ++i) 
    {
        if (i % (Nx*Ny) == 0)
            std::cout << "Reached end of layer" << '\n';
        std::cout << er[i] << '\n';
    }
    */
    // convMat, Toeplitz
    std::cout << "Size of erc " << h_erc.size() << '\n'; /////////////////// To Be checked

    for (size_t i = 0; i < PQ*PQ*thickness.size(); ++i) 
    {
        if (i < (PQ*PQ))
            std::cout << h_erc[i] << '\n';
    }

    
    return 0;
}