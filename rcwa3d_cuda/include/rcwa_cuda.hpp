#pragma once

#include <cufft.h>
#include <vector>
#include <complex>
using namespace std::complex_literals; // Enables the '1i' literal
using Real = double;
using Complex = std::complex<Real>;
using cuda_Complex_d = cufftDoubleComplex; // cuFFT Z2Z
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
    // 3D arrays: er[iy][ix][layer] stored as flat vector er[layer * Nx * Ny + iy * Nx + ix] (row-major order)
    // spatial arrays
    std::vector<Complex> er{};  // permittivity Ny × Nx x num_layers
    std::vector<Complex> ur{};  // permeability Ny × Nx x num_layers
    std::vector<Real> t{};   // thickness per layer, size num_layers

    Device(int Nx_, int Ny_, int num_layers_, Real Lx_, Real Ly_, const std::vector<Complex>& er_, 
        const std::vector<Complex>& ur_, const std::vector<Real>& t_, int Nx_harmonics_, int Ny_harmonics_) 
        : Nx(Nx_), Ny(Ny_), num_layers(num_layers_), Lx(Lx_), Ly(Ly_), er(er_), ur(ur_), t(t_)
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

// Funtion that defines the normalization kernel for the FFT data
__global__ void normalizeFFT(cuda_Complex_d* data, int Nx, int Ny, int num_layers);


// Utility functions
// ======================================================================================

namespace layer
{
    inline std::vector<Complex> uniform(int Nx, int Ny, Complex value)
    {
        // Function that creates a uniform 2D array of size Nx x Ny with a given complex value
        return std::vector<Complex>(Nx * Ny, value);

    }


    inline std::vector<Complex> x_pattern(int Nx, int Ny, double Lx, double Ly)
    {
        std::vector<Complex> out(Nx * Ny, Complex(0.0, 0.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
        double y = (iy + 0.5) * Ly / Ny;  // pyhsical [0, Ly]
        for (int ix = 0; ix < Nx; ++ix)
            {
                double x = (ix + 0.5) * Lx / Nx;  // pyhsical [0, Lx]
                bool inside_square = (std::abs(x - Lx/2.0) <= 0.25);
                out[iy * Nx + ix] = inside_square ? Complex(2.0, 0.0) : Complex(8.0, 0.0);
            }
        }
        return out;

    }


    inline std::vector<Complex> y_pattern(int Nx, int Ny, double Lx, double Ly)
    {
        std::vector<Complex> out(Nx * Ny, Complex(0.0, 0.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
        double y = (iy + 0.5) * Ly / Ny;  // pyhsical [0, Ly]
        for (int ix = 0; ix < Nx; ++ix)
            {
                double x = (ix + 0.5) * Lx / Nx;  // pyhsical [0, Lx]
                bool inside_square = (std::abs(y - Ly/2.0) <= 0.25);
                out[iy * Nx + ix] = inside_square ? Complex(2.0, 0.0) : Complex(8.0, 0.0);
            }
        }
        return out;

    }


    inline std::vector<Complex> triangle_pattern(int Nx, int Ny, double Lx, double Ly)
    {
        std::vector<Complex> out(Nx * Ny, Complex(0.0, 0.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
            double y = (iy + 0.5) * Ly / Ny;  // physical y coordinate
            for (int ix = 0; ix < Nx; ++ix)
                {
                    double x = (ix + 0.5) * Lx / Nx;  // physical x coordinate
                    // Triangle centred at (Lx/2, Ly/2) with base w = 0.7*Ly
                    double w = 0.70 * Ly;
                    double xc = Lx / 2.0;
                    double yc = Ly / 2.0;
                    // Triangle vertices (centred)
                    double y_rel = y - (yc - w/2.0);  // relative to triangle base
                    double half_base = (w/2.0) * (y_rel/w);  // apex at top, base at bottom                
                    bool inside_triangle = (y_rel >= 0) && (y_rel <= w) && (std::abs(x - xc) <= half_base);
                    out[iy * Nx + ix] = inside_triangle ? Complex(2.0, 0.0) : Complex(6.0, 0.0);
                }
        }
        return out;

    }


    inline std::vector<Complex> square_pattern(int Nx, int Ny, Complex background_val, Complex square_val) 
    {
    
        // Allocate the single 1D vector representing the entire Ny x Nx grid
        std::vector<Complex> out(Ny * Nx, background_val);

        // Define center and half-width of the square
        int cy = Ny / 2;
        int cx = Nx / 2;
        int half_size = std::min(Nx, Ny) / 4; // Controls the size of the square

        // Calculate boundary indices
        int y_start = cy - half_size;
        int y_end   = cy + half_size;
        int x_start = cx - half_size;
        int x_end   = cx + half_size;

        // Fill the square values within boundaries
        for (int y = y_start; y < y_end; ++y) {
            for (int x = x_start; x < x_end; ++x) {
                if (y >= 0 && y < Ny && x >= 0 && x < Nx) {
                    // Flattened 1D index mapping for (y, x) row-major order
                    out[y * Nx + x] = square_val;
                }
            }
        }
        return out;

    }


    inline std::vector<Complex> stack(const std::vector<std::vector<Complex>>& layers)
    {
        size_t total = 0;
        for (const auto& layer : layers)
            total += layer.size();

        std::vector<Complex> out;
        out.reserve(total); // reserve the memory for the total size of all layers

        for (const auto& layer : layers)
            out.insert(out.end(), layer.begin(), layer.end());

        return out;

    }
}
