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
            layer::triangle_pattern(Nx, Ny, Lx, Ly, Complex(2.0, 0.0), Complex(6.0, 0.0)),
            layer::x_pattern(Nx, Ny, Lx, Ly, Complex(2.0, 0.0), Complex(6.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::triangle_pattern(Nx, Ny, Lx, Ly, Complex(2.0, 0.0), Complex(6.0, 0.0))


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
            layer::triangle_pattern_shifted(Nx, Ny, Lx, Ly, 0.35, 0.35, Complex(1.0, 0.0), Complex(5.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::triangle_pattern_shifted(Nx, Ny, Lx, Ly, 0.2, 0.2, Complex(2.0, 0.0), Complex(8.0, 0.0))


    };
    std::vector<Complex> er = layer::stack(er_layers);
    std::vector<Complex> ur = layer::stack(ur_layers);

    //std::vector<Real> thickness{1.5, 0.3, 2.4, 2.6, 1.1, 0.25};
    std::vector<Real> thickness{1.5, 0.3, 2.4, 3.0};

    // Build the device --> for now materials are not wavelength dependent
    Device device(Nx, Ny, thickness.size(), Lx, Ly, er, ur, thickness, Nx_harmonics, Ny_harmonics);

    // Build reflection and transmission regions --> for now the materials are non wavelength dependent, extension to wavelength dependent materials is possible
    Complex er_ref(2.0, 0.0);
    Complex ur_ref(1.0, 0.0);
    Complex er_trn(9.0, 0.0);
    Complex ur_trn(1.0, 0.0);
    RCWAParams params(Nx_harmonics, Ny_harmonics, er_ref, ur_ref, er_trn, ur_trn);

    // Build all the sources that we will simulate
    size_t num_wavelengths = 100;  // # of sources to simulate

    std::vector<Real> lambda0_v(num_wavelengths);
    std::vector<Real> k0_v(num_wavelengths);
    Real lambda_start = 1.0;
    Real lambda_end = 2.0;
    // If num_wavelengths is 1, just set it to the lambda_start value to avoid division by zero
    if (num_wavelengths == 1)
    {
        lambda0_v[0] = lambda_start;
        k0_v[0] = 2 * M_PI / lambda0_v[0];
    }
    else 
    {
        Real step = (lambda_end - lambda_start) / (num_wavelengths - 1);
        for (int i = 0; i < num_wavelengths; ++i) 
        {
            lambda0_v[i] = lambda_start + i * step;
            k0_v[i] = 2 * M_PI / lambda0_v[i];
        }
    }

    Real theta = M_PI / 6.0;
    std::vector<Real> theta_v(num_wavelengths, theta); // All thetas are the same for now
    Real phi = M_PI / 12.0;
    std::vector<Real> phi_v(num_wavelengths, phi); // All phis are the same for now
    Real pte = 0.0; // TE pol
    std::vector<Real> pte_v(num_wavelengths, pte); // All ptes are the same for now
    Real ptm = 1.0; // TM pol
    std::vector<Real> ptm_v(num_wavelengths, ptm); // All ptms are the same for now
    Source source(lambda0_v, k0_v, theta_v, phi_v, pte_v, ptm_v);



    // CUDA

    int num_gpus = 0;
    CHECK(cudaGetDeviceCount(&num_gpus));
    for (int g = 0; g < num_gpus; ++g)
    {
        CHECK(cudaSetDevice(g));
        CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)); // waiting threads sleep instead of spinning
    }


    std::atomic<int> next_wavelength{0};
    StreamWorker worker;
    InitWorker(worker, 0, device, params, /*debug=*/true);
    RunWorker(worker, device, source, params, next_wavelength); // will be moved to std::thread
    
    // Convolution results
    const auto* erc = reinterpret_cast<const Complex*>(worker.h_erc); // raw pointer
    const auto* urc = reinterpret_cast<const Complex*>(worker.h_urc);



    //////////////////////////////////////////////////
    // TESTING
    //////////////////////////////////////////////////
    // CHECK WILL FAIL HERE
    // void* p = nullptr;
    // CHECK(cudaMalloc(&p, size_t(1) << 50)); // 1 PB


    // fft
    /*
    std::cout << "Size of er " << er.size() << '\n';

    for (size_t i = 0; i < Nx*Ny*thickness.size(); ++i) 
    {
        if (i % (Nx*Ny) == 0)
            std::cout << "Reached end of layer" << '\n';
        std::cout << er[i] << '\n';
    }
    */
    // convMat, Toeplitz
    const size_t convmat_elements = static_cast<size_t>(PQ) * PQ * device.num_layers;
    std::cout << "Size of erc " << convmat_elements << '\n';

    for (size_t i = 0; i < PQ*PQ*thickness.size(); ++i) 
    {
        if (i >= 3*(PQ*PQ) && i < 4*(PQ*PQ))
            std::cout << erc[i] << '\n';
    }
    std::cout << "----------------------------------------" << '\n';
    std::cout << "Size of urc " << convmat_elements << '\n'; 

    for (size_t i = 0; i < PQ*PQ*thickness.size(); ++i) 
    {
        if (i >= 3*(PQ*PQ) && i < 4*(PQ*PQ))
            std::cout << urc[i] << '\n';
    }

    DestroyWorker(worker);
    
    return 0;
}