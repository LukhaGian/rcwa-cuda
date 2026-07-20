#include "rcwa.hpp"
#include <iostream>

int main()
{
    int Nx = 16; // make sure it's always even
    int Ny = 16;

    int Nx_harmonics = 1;
    int Ny_harmonics = 2;

    std::vector<Real> er(5);
    std::vector<Real> ur(5);
    std::vector<Real> t(5);
    Device device(Nx, Ny, 1, 1.0, 1.0, er, ur, t, Nx_harmonics, Ny_harmonics);
    Source source(1, M_PI/4, M_PI/4, 1, 1);
    //Source source(1, 0.0, 0.0, 1, 1);
    Complex er_ref(1.0, 0.0);
    Complex ur_ref(1.0, 0.0);
    //Complex er_ref(6.0, -0.13);
    //Complex ur_ref(3.5, -0.15);
    Complex er_trn(1.0, 0.0);
    Complex ur_trn(1.0, 0.0);
    RCWAParams params(Nx_harmonics, Ny_harmonics, er_ref, ur_ref, er_trn, ur_trn);
    //std::cout << params.er_ref << '\n';
    std::vector<Complex> k_inc(3);
    Vector Kx, Ky, Kz_ref, Kz_trn;
    ComputeWaveVectors(device, source, params, k_inc, Kx, Ky, Kz_ref, Kz_trn);
    Matrix W0, V0(2*(2*Nx_harmonics+1)*(2*Ny_harmonics +1), 2*(2*Nx_harmonics+1)*(2*Ny_harmonics +1));
    GapMedium(Kx, Ky, W0, V0);

    Complex er_val(4.0, 0.0);
    std::vector<Complex> field(Nx * Ny, er_val); // uniform field of size Nx * Ny

    Matrix C = ConvMat(field, 0, Nx, Ny, Nx_harmonics, Ny_harmonics);
    std::cout << C << '\n';
    //MatrixXcd C = MatrixXcd::Zero(Nx, Ny);
    std::cout << "Size of C: " << C.rows() << " x " << C.cols() << std::endl;
    ScatteringMatrix A(C, C, C, C);
    ScatteringMatrix B(C, C, C, C);
    ScatteringMatrix res = RedhefferProduct(A, B);
    std::complex<double> a(-1.0, 0.0);
    std::cout << a*a << '\n';
    std::complex<double> b(-1.0, -0.0);
    if (a == b)
        std::cout << "Gotcha" << '\n';
    if (unsigned_sqrt(a) == unsigned_sqrt(b))
        std::cout << "Gotcha2" << '\n';
    if (std::sqrt(a) == std::sqrt(b))
        std::cout << "Gotcha2.1" << '\n';    
    if (a*a == b*b)
        std::cout << a*a <<'\n';
        std::cout << b*b << '\n';
        std::cout << "Gotcha3" << '\n';     
    std::cout << std::sqrt(a) << " " << std::sqrt(b) << '\n';      
    /*
    std::complex<double> a(-1.0, 0.0);
    Eigen::MatrixXcd mat(2, 2);
    mat << a,  std::complex<double>(3, 4),
       std::complex<double>(-1, 0), std::complex<double>(0, -1);
    std::cout << mat.array().sqrt().conjugate() << '\n';
    std::cout << std::conj(std::sqrt(a)) <<'\n';
    std::complex<double> z(0.0, -1.0);
    std::cout << z*z << '\n';
    */

    /*
    // Meshgrid testing
    Vector k_x_tilde = Vector::LinSpaced(2*params.Nx_harmonics + 1, -params.Nx_harmonics, params.Nx_harmonics);
    std::cout << k_x_tilde << '\n';
    Vector k_y_tilde = Vector::LinSpaced(2*params.Ny_harmonics + 1, -params.Ny_harmonics, params.Ny_harmonics);
    std::cout << k_y_tilde << '\n';
    Matrix Kx_tilde = Matrix::Zero(k_y_tilde.size(), k_x_tilde.size());
    Matrix Ky_tilde = Matrix::Zero(k_y_tilde.size(), k_x_tilde.size());
    MeshGrid(k_x_tilde, k_y_tilde, Kx_tilde, Ky_tilde);
    std::cout << Kx_tilde << '\n';
    std::cout << Ky_tilde << '\n';
    */
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