#pragma once

#include <vector>
#include <complex>
#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>

using namespace std::complex_literals; // Enables the '1i' literal
using Real = double;
using Complex = std::complex<Real>;
using Matrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>; // it is a complex matrix with dynamic size (by default column-major)
using Vector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using Real_Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>; // real matrix
using Real_Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>; // real vector
using Vec_3d = Eigen::Matrix<Complex, 3, 1>; // 3d complex vector


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
    // convolution space arrays
    std::vector<Matrix> erc{}; // convolution of perimittivity er num_layers x PQ × PQ
    std::vector<Matrix> urc{}; // convolution of permeability ur num_layers x PQ × PQ

    Device(int Nx_, int Ny_, int num_layers_, Real Lx_, Real Ly_, const std::vector<Complex>& er_, 
        const std::vector<Complex>& ur_, const std::vector<Real>& t_, int Nx_harmonics_, int Ny_harmonics_) 
        : Nx(Nx_), Ny(Ny_), num_layers(num_layers_), Lx(Lx_), Ly(Ly_), er(er_), ur(ur_), t(t_),
        erc(num_layers_, Matrix::Zero((2 * Ny_harmonics_ + 1) * (2 * Nx_harmonics_ + 1), (2 * Ny_harmonics_ + 1) * (2 * Nx_harmonics_ + 1))),
        urc(num_layers_, Matrix::Zero((2 * Ny_harmonics_ + 1) * (2 * Nx_harmonics_ + 1), (2 * Ny_harmonics_ + 1) * (2 * Nx_harmonics_ + 1)))
    {
    }
};

// struct that defines the Source properties
struct Source
{
    Real lambda0{}; // wavelength in vacuum
    Real k0{}; // wave number in vacuum = 2pi/lambda0
    Real theta{}; // polar angle of incidence
    Real phi{}; // azimuthal angle of incidence
    Real pte{}; // TE polarisation amplitude
    Real ptm{}; // TM polarisation amplitude

    Source(Real lambda0_, Real theta_, Real phi_, Real pte_, Real ptm_) 
    : lambda0(lambda0_), k0(2 * M_PI / lambda0_), theta(theta_), phi(phi_), pte(pte_), ptm(ptm_)
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


// struct that defines the RCWA results, i.e. reflection and transmission
struct Results
{
    Real_Matrix R{}; // reflection coefficients for each diffraction order
    Real_Matrix T{}; // transmission coefficients for each diffraction order
    Real R_tot{}; // total reflection
    Real T_tot{}; // total transmission
    // Note: for energy conservation, R_tot + T_tot = 1

    Results(Real_Matrix R_, Real_Matrix T_, Real R_tot_, Real T_tot_)
    : R(R_), T(T_), R_tot(R_tot_), T_tot(T_tot_)
    {
    }
};


// struct that contains a Scattering Matrix
struct ScatteringMatrix
{
    // S-matrix blocks for each layer
    Matrix S11{}; // reflection from the left
    Matrix S12{}; // transmission from left to right
    Matrix S21{}; // transmission from right to left
    Matrix S22{}; // reflection from the right
    
    ScatteringMatrix(Matrix S11_, Matrix S12_, Matrix S21_, Matrix S22_)
    : S11(S11_), S12(S12_), S21(S21_), S22(S22_)
    {
    }
};


//Compute the Convolution matrices for each layer and each field (er, ur)
Matrix ConvMat(const std::vector<Complex>& field, int layer, int Nx, int Ny, int Nx_harmonics, int Ny_harmonics);

// Compute Wave Vector Expansion, ie update k_inc, Kx, Ky, Kz_ref, Kz_trn
void ComputeWaveVectors(const Device& device, const Source& source, const RCWAParams& params, std::vector<Complex>& k_inc, Vector& Kx, Vector& Ky, Vector& Kz_ref, Vector& Kz_trn);

// Compute eigenmodes of Gap medium, ie compute W0 and V0 for the gap medium
void GapMedium(const Vector& Kx, const Vector& Ky, Matrix& W0, Matrix& V0);

// Initialize the Device Scattering Matrix, ie return the initial Scattering Matrix S_device
ScatteringMatrix SMatrixInit(int Nx_harmonics, int Ny_harmonics);

// Iteration per layer (build eigen value problem per layer), compute the S-matrix for each layer, and combine them using Redheffer product wrt S_device
ScatteringMatrix SMatrixLayer(int layer, const Device& device, const Source& source, const RCWAParams& params, const Vector& Kx, const Vector& Ky, const Matrix& W0, const Matrix& V0);

// Compute Reflection Side Connection SMatrix, ie S_ref
ScatteringMatrix SMatrixReflection(const RCWAParams& params, const Vector& Kx, const Vector& Ky, Vector& Kz_ref, const Matrix& W0, const Matrix& V0, Matrix& W_ref);

// Compute Transmission Side Connection SMatrix, ie S_trn
ScatteringMatrix SMatrixTransmission(const RCWAParams& params, const Vector& Kx, const Vector& Ky, Vector& Kz_trn, const Matrix& W0, const Matrix& V0, Matrix& W_trn);

// Compute Mode coefficients of the Source
void ComputeSourceModeCoeff(const Source& source, const RCWAParams& params, const std::vector<Complex>& k_inc, const Matrix& W_ref, Vector& csrc);

// Compute Reflected and Transmitted Fields
void ComputeReflectedField(const RCWAParams& params, const ScatteringMatrix& S_global, const Vector& csrc, const Vector& Kx, const Vector& Ky, const Vector& Kz_ref, const Matrix& W_ref, Vector& r); // reflected field r
void ComputeTransmittedField(const RCWAParams& params, const ScatteringMatrix& S_global, const Vector& csrc, const Vector& Kx, const Vector& Ky, const Vector& Kz_trn, const Matrix& W_trn, Vector& t); // transmitted field t

// Compute Diffraction Efficiencies
Results ComputeDiffractionEfficiencies(const RCWAParams& params, const Vector& r, const Vector& t, const std::vector<Complex>& k_inc, const Vector& Kz_ref, const Vector& Kz_trn);


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


// Redheffer Product A x B
ScatteringMatrix RedhefferProduct(const ScatteringMatrix& A, const ScatteringMatrix& B); 


inline Complex unsigned_sqrt(Complex z)
{
    /*
    Safe complex square root, it sanitises -0 imaginary part before sqrt
    to avoid branch cut issues from IEEE 754 signed zero
    */
    if (z.imag() == 0.0) z = Complex(z.real(), 0.0);
    return std::sqrt(z);
    
}


template<typename Derived>
inline auto unsigned_sqrt(const Eigen::EigenBase<Derived>& expr)
{
    /*
    Eigen expression version: it accepts Matrix, Vector, Array and lazy expressions.
    It returns an expression that computes the unsigned square root of each element in the input expression.
    */
    return expr.derived().unaryExpr([](Complex z) -> Complex {
        if (z.imag() == 0.0) z = Complex(z.real(), 0.0);
        return std::sqrt(z);
    });
}


// MeshGrid function, template
template <typename Scalar>
void MeshGrid(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y,
              Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Y)
{
    // Create meshgrid matrices X and Y from vectors x and y
    X = x.transpose().replicate(y.size(), 1);
    Y = y.replicate(1, x.size());
}


struct EigenvalSolverResults
{
    // Struct to hold the results of the eigenvalue solver
    Vector eigenvalues{};
    Matrix eigenvectors{};
};

EigenvalSolverResults extraction(Eigen::ComplexEigenSolver<Matrix>&& solver); // Function to extract eigenvalues and eigenvectors from a ComplexEigenSolver object, using move semantics to avoid unnecessary copies

// ======================================================================================
