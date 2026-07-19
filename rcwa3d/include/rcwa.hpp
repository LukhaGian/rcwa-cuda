#pragma once

#include <vector>
#include <complex>
#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <cmath>
//#include <iostream>

using namespace std::complex_literals; // Enables the '1i' literal
using Real = double;
using Complex = std::complex<Real>;
using Matrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>; // check properties, it is a complex matrix with dynamic size (BY DEFAULT COLUMN MAJOR ORDER)
using Vector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;

using MatrixXcd = Eigen::MatrixXcd; 
using VectorXcd = Eigen::VectorXcd;



// struct that defines the Device properties
struct Device 
{
    // grid resolution per layer
    int Nx{}; // spatial grid points along X axis (columns)
    int Ny{}; // spatial grid points along Y axis (rows)
    int num_layers{};  // number of layers M
    Real Lx{};        // unit cell dimensions
    Real Ly{};
    // 3D arrays: er[iy][ix][layer] stored as flat vector er[layer * Nx * Ny + iy * Nx + ix] (row-major order) --> be careful and TBD

    // spatial arrays
    std::vector<Real> er{};  // permittivity Ny × Nx x num_layers
    std::vector<Real> ur{};  // permeability Ny × Nx x num_layers
    std::vector<Real> t{};   // thickness per layer, size num_layers

    // convolution space arrays: TBD if 3d Tensor solution is better or std::vector<Matrix> suffices (iterating over layers)
    std::vector<Matrix> erc{}; // convolution of perimittivity er num_layers x Ny_harmonics × Nx_harmonics
    std::vector<Matrix> urc{}; // convolution of permeability ur num_layers x Ny_harmonics × Nx_harmonics

    Device() : Nx(1), Ny(1), num_layers(1), Lx(1.0), Ly(1.0) // Default constructor, add the remaining default entries for the field DONT USE IT FOR NOW
    {
    }

    Device(int Nx_, int Ny_, int num_layers_, Real Lx_, Real Ly_, const std::vector<Real>& er_, 
        const std::vector<Real>& ur_, const std::vector<Real>& t_, int Nx_harmonics_, int Ny_harmonics_) 
        : Nx(Nx_), Ny(Ny_), num_layers(num_layers_), Lx(Lx_), Ly(Ly_), er(er_), ur(ur_), t(t_),
        erc(num_layers_, Matrix::Zero(Ny_harmonics_, Nx_harmonics_)), urc(num_layers_, Matrix::Zero(Ny_harmonics_, Nx_harmonics_))
    {
    }
};

// struct that defines the Source properties
struct Source
{
    Real lambda0{}; // wavelength in vacuum
    Real k0{};    // wave number in vacuum = 2pi/lambda0
    Real theta{}; // polar angle of incidence
    Real phi{}; // azimuthal angle of incidence
    //
    Real pte{};       // TE polarisation amplitude
    Real ptm{};       // TM polarisation amplitude

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
    // Support for non lossless materials
    Complex er_ref{};    // permittivity of reflection region (superstrate)
    Complex ur_ref{};    // permeability of reflection region
    Complex er_trn{};    // permittivity of transmission region (substrate)
    Complex ur_trn{};    // permeability of transmission region

    RCWAParams(int Nx_harmonics_, int Ny_harmonics_, Complex er_ref_, Complex ur_ref_, Complex er_trn_, Complex ur_trn_)
    : Nx_harmonics(Nx_harmonics_), Ny_harmonics(Ny_harmonics_), er_ref(er_ref_), ur_ref(ur_ref_), er_trn(er_trn_), ur_trn(ur_trn_)
    {
    }
};


// struct that defines the RCWA results, i.e. reflection and transmission
struct Results
{
    std::vector<Real> R{}; // reflection coefficients for each diffraction order
    std::vector<Real> T{}; // transmission coefficients for each diffraction order
    Real R_tot{}; // total reflection
    Real T_tot{}; // total transmission
    // Note: for energy conservation, R_tot + T_tot = 1
};


// struct that contains a Scattering Matrix
struct ScatteringMatrix
{
    // S-matrix blocks for each layer
    Matrix S11{}; // reflection from the left
    Matrix S12{}; // transmission from left to right
    Matrix S21{}; // transmission from right to left
    Matrix S22{}; // reflection from the right
    
    ScatteringMatrix(Matrix S11_, Matrix S12_, Matrix S21_, Matrix S22_) : S11(S11_), S12(S12_), S21(S21_), S22(S22_)
    {
    }
    /*
    ScatteringMatrix() = default;

    ScatteringMatrix(const Matrix& a, const Matrix& b,
                     const Matrix& c, const Matrix& d)
        : S11(a), S12(b), S21(c), S22(d) {
            std::cout << "constructor\n";
        }

    ScatteringMatrix(const ScatteringMatrix& other) : S11(other.S11), S12(other.S12), S21(other.S21), S22(other.S22) {
        std::cout << "copy\n";
    }

    ScatteringMatrix(ScatteringMatrix&& other) noexcept
        : S11(std::move(other.S11)), S12(std::move(other.S12)),
          S21(std::move(other.S21)), S22(std::move(other.S22)) {
        std::cout << "move\n";
    }
    */
};


// Computation flow

// 1. Initialization

// 2. Build device layers on grid (initialize the Device)

// 3. Compute the Convolution matrices for each layer and each field (er, ur)
Matrix ConvMat(const std::vector<Real>& field, int layer, int Nx, int Ny, int Nx_harmonics, int Ny_harmonics); // add then in place the convolution matrices for each layer and each field (er, ur) into the Device struct

// 4. Compute Wave Vector Expansion, ie update k_inc, Kx, Ky, Kz_ref, Kz_trn
void ComputeWaveVectors(const Device& device, const Source& source, const RCWAParams& params, std::vector<Complex>& k_inc, Vector& Kx, Vector& Ky, Vector& Kz_ref, Vector& Kz_trn);

// 5. Compute eigenmodes of Gap medium, ie compute W0 and V0 for the gap medium
void GapMedium(const Vector& Kx, const Vector& Ky, Matrix& W0, Matrix& V0);

// 6. Initialize the Device Scattering Matrix, ie return the initial Scattering Matrix S_device
ScatteringMatrix SMatrixInit(int Nx_harmonics, int Ny_harmonics);

// 7. Iteration per layer (build eigen value problem per layer), compute the S-matrix for each layer, and combine them using Redheffer product wrt S_device
ScatteringMatrix SMatrixLayer(int layer, const Device& device, const Source& source, const RCWAParams& params, const Vector& Kx, const Vector& Ky, const Matrix& W0, const Matrix& V0);
// remember to .resize(0,0) to free the memory when the S-matrix is no longer needed, e.g. after Redheffer product, OR use blocks in main for sequential

// Update Device matrix

// 8. Compute Reflection Side Connection SMatrix, ie S_ref
ScatteringMatrix SMatrixReflection(const RCWAParams& params, const Vector& Kx, const Vector& Ky, Vector& Kz_ref, const Matrix& W0, const Matrix& V0);

// 9. Compute Transmission Side Connection SMatrix, ie S_trn
ScatteringMatrix SMatrixTransmission(const RCWAParams& params, const Vector& Kx, const Vector& Ky, Vector& Kz_trn, const Matrix& W0, const Matrix& V0);

// 10. Compute Global SMatrix, S_global = S_ref x S_device x S_trn
// call RedhefferProduct() to combine the S-matrices of ref + device + trn into global

// 11. Compute Source parameters (source field)
// compute esrc and csrc 
void ComputeSourceField(const Source& source, const RCWAParams& params, const std::vector<Complex>& k_inc, std::vector<Complex>& esrc, std::vector<Complex>& csrc);

// 12. Compute Reflected and Transmitted Fields
// reflected field r
std::vector<Complex> ComputeReflectedField(const ScatteringMatrix& S_global, const std::vector<Complex>& csrc, const Matrix& Kx, const Matrix& Ky, const Matrix& Kz_ref); // W_ref is considered = I
// transmitted field t
std::vector<Complex> ComputeTransmittedField(const ScatteringMatrix& S_global, const std::vector<Complex>& csrc, const Matrix& Kx, const Matrix& Ky, const Matrix& Kz_trn); // W_trn is considered = I

// 13. Compute Diffraction Efficiencies
Results ComputeDiffractionEfficiencies(const RCWAParams& params, std::vector<Complex> r, std::vector<Complex> t, const std::vector<Complex>& k_inc, const Matrix& Kz_ref, const Matrix& Kz_trn);

// 14. Output results
void OutputResults(const Results& results, const RCWAParams& params);

// MOVE TO ADDITIONAL?
// Redheffer Product A x B
ScatteringMatrix RedhefferProduct(const ScatteringMatrix& A, const ScatteringMatrix& B); 
// remember to use std::move() to avoid copying the resulting S-matrix in place AND .resize(0,0) to free the memory when the S-matrix is no longer needed, e.g. after Redheffer product

// MeshGrid function, template
template <typename Scalar>
void MeshGrid(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y,
              Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Y)
{
    X = x.transpose().replicate(y.size(), 1);
    Y = y.replicate(1, x.size());
}

// conj_unsigned_zero function, template
template <typename T>
std::complex<T> conj_unsigned_zero(const std::complex<T>& z) {
    // Function that computes the conjugate of a complex number, avoiding IEEE 754 standard for std::conj()
    // if z = (1.0, 0.0) ==> std::conj(z) = (1.0, -0.0)
    // If the imaginary part is exactly 0 (positive or negative), keep it 0.0
    // Otherwise, negate it normally.
    T imag_part = (z.imag() == T(0.0)) ? T(0.0) : -z.imag();
    return {z.real(), imag_part};
}

struct EigenvalSolverResults
{
    Vector eigenvalues{};
    Matrix eigenvectors{};
};

EigenvalSolverResults extraction(Eigen::ComplexEigenSolver<Matrix>&& solver);
