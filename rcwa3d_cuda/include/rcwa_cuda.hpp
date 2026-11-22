#pragma once

#include "rcwa_layers.hpp"
#include "cuda_check.hpp"
#include <cuComplex.h> // for make_cuDoubleComplex
#include <cufft.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <complex>
using namespace std::complex_literals; // Enables the '1i' literal

using cuda_Complex_d = cufftDoubleComplex; // cuFFT Z2Z
static_assert(sizeof(Complex) == sizeof(cuda_Complex_d), "Complex and cuda_Complex_d must have the same size"); // explicit check, safe
using cuda_Real_d = cufftDoubleReal; // cuFFT Real

// struct that defines the Device properties
struct Device 
{
    // grid resolution per layer
    int Nx{}; // spatial grid points along X axis (columns)
    int Ny{}; // spatial grid points along Y axis (rows)
    int num_layers{}; // number of layers M
    Real Lx{}; // unit cell dimension along X axis
    Real Ly{}; // unit cell dimension along Y axis
    std::vector<ShapeDesc> er_desc{};  // vector containing the shape descriptions for each layer in er/ur
    std::vector<ShapeDesc> ur_desc{};
    std::vector<Real> t{};   // thickness per layer, size num_layers

    Device(int Nx_, int Ny_, int num_layers_, Real Lx_, Real Ly_, const std::vector<ShapeDesc>& er_desc_, 
        const std::vector<ShapeDesc>& ur_desc_, const std::vector<Real>& t_, int Nx_harmonics_, int Ny_harmonics_) 
        : Nx(Nx_), Ny(Ny_), num_layers(num_layers_), Lx(Lx_), Ly(Ly_), er_desc(er_desc_), ur_desc(ur_desc_), t(t_)
    {
    }
};

// struct that defines the Source properties 
struct Source
{
    std::vector<Real> lambda0{}; // wavelength in vacuum
    std::vector<Real> k0{}; // wave number in vacuum = 2pi/lambda0
    std::vector<Real> theta{}; // polar angle of incidence
    std::vector<Real> phi{}; // azimuthal angle of incidence
    std::vector<Real> pte{}; // TE polarisation amplitude
    std::vector<Real> ptm{}; // TM polarisation amplitude

    Source(const std::vector<Real> lambda0_, const std::vector<Real> k0_, const std::vector<Real> theta_, const std::vector<Real> phi_, const std::vector<Real> pte_, const std::vector<Real> ptm_) 
    : lambda0(lambda0_), k0(k0_), theta(theta_), phi(phi_), pte(pte_), ptm(ptm_)
    {
    }
};

// struct that defines the RCWA simulation parameters
struct RCWAParams
{
    int Nx_harmonics{}; // number of harmonics along X axis
    int Ny_harmonics{}; // number of harmonics along Y axis
    Complex er_ref{}; // permittivity of reflection region (superstrate)
    Complex ur_ref{}; // permeability of reflection region
    Complex er_trn{}; // permittivity of transmission region (substrate)
    Complex ur_trn{}; // permeability of transmission region

    RCWAParams(int Nx_harmonics_, int Ny_harmonics_, Complex er_ref_, Complex ur_ref_, Complex er_trn_, Complex ur_trn_)
    : Nx_harmonics(Nx_harmonics_), Ny_harmonics(Ny_harmonics_), er_ref(er_ref_), ur_ref(ur_ref_), er_trn(er_trn_), ur_trn(ur_trn_)
    {
    }
};


// Build layers
__global__ void BuildLayers(const ShapeDesc* er_desc, const ShapeDesc* ur_desc, const cuda_Complex_d* values,
                            cuda_Complex_d* er, cuda_Complex_d* ur, int Nx, int Ny, double Lx, double Ly);

// Function that performs FFT and normalization on the device
void convMat_cu(cufftHandle& plan, cuda_Complex_d* d_er_ptr, cuda_Complex_d* d_ur_ptr, cuda_Complex_d* d_erc_ptr,
                cuda_Complex_d* d_urc_ptr, int Nx, int Ny, int num_layers, int Nx_harmonics, int Ny_harmonics, int Q, int PQ, cudaStream_t stream);

// Kernel that defines the normalization for the FFT data and swaps opposite FFT quadrants for each layer
__global__ void NormalizeQuadShiftFFT(cuda_Complex_d* data, int Nx, int Ny, int num_layers);

// Kernel that assembles the convolution matrix
__global__ void convMatAssembly(cuda_Complex_d* fft_data, cuda_Complex_d* conv_data, int Nx, int Ny, int num_layers, int Nx_harmonics, int Ny_harmonics, int Q, int PQ);



