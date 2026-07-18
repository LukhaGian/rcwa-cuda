#include "rcwa.hpp"
#include <iostream>

int main()
{
    int Nx = 10;
    int Ny = 10;

    int Nx_harmonics = 1;
    int Ny_harmonics = 1;

    std::vector<Real> er(5);
    std::vector<Real> ur(5);
    std::vector<Real> t(5);
    Device device(Nx, Ny, 1, 1.0, 1.0, er, ur, t, Nx_harmonics, Ny_harmonics);
    //Source source(1, M_PI/4, M_PI/4, 1, 1);
    Source source(1, 0.0, 0.0, 1, 1);

    RCWAParams params(Nx_harmonics, Ny_harmonics, 1, 1, 1, 1);
    std::vector<Complex> k_inc(3);
    Vector Kx, Ky, Kz_ref, Kz_trn;
    ComputeWaveVectors(device, source, params, k_inc, Kx, Ky, Kz_ref, Kz_trn);
    Matrix W0, V0(2*(2*Nx_harmonics+1)*(2*Ny_harmonics +1), 2*(2*Nx_harmonics+1)*(2*Ny_harmonics +1));
    GapMedium(Kx, Ky, W0, V0);

    Real er_val = 4.0;
    std::vector<Real> field(Nx * Ny, er_val); // uniform field of size Nx * Ny

    Matrix C = ConvMat(field, 0, Nx, Ny, Nx_harmonics, Ny_harmonics);
    //MatrixXcd C = MatrixXcd::Zero(Nx, Ny);
    //std::cout << C << '\n';
    //std::cout << "Size of C: " << C.rows() << " x " << C.cols() << std::endl;
    ScatteringMatrix A(C, C, C, C);
    ScatteringMatrix B(C, C, C, C);
    ScatteringMatrix res = RedhefferProduct(A, B);

    /*
    Eigen::Matrix2d Z;
    Z << 0,-1,1,0;

    // Compute eigenvalues and eigenvectors
    Eigen::EigenSolver<Eigen::Matrix2d> solver(Z);

    if (solver.info() != Eigen::Success) abort();

    // Output complex results
    std::cout << "Eigenvalues:\n" << solver.eigenvalues() << "\n\n";
    std::cout << "Eigenvectors:\n" << solver.eigenvectors() << "\n";
    */

    return 0;
}