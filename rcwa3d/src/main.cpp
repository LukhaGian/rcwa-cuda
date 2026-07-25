#include "rcwa.hpp"
#include <iostream>
#include <algorithm>

// Generates a single flattened vector of size Ny * Nx with a central square
std::vector<Complex> createSquareMatrix1D(int Ny, int Nx, Complex background_val, Complex square_val) {
    
    // Allocate the single 1D vector representing the entire Ny x Nx grid
    std::vector<Complex> matrix(Ny * Nx, background_val);

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
                matrix[y * Nx + x] = square_val;
            }
        }
    }

    return matrix;
}

int main()
{
    int Nx = 1024; // make sure it's always even
    int Ny = 1024;

    int Nx_harmonics = 1;
    int Ny_harmonics = 1;

    Complex add_val (3.5, 0.0);
    std::vector<Complex> add_field(Nx * Ny, add_val);
    Complex square_val (4.5, 0.0);
    std::vector<Complex> squared_field = createSquareMatrix1D(Ny, Nx, add_val, square_val);
    Complex er_val(4.0, 0.0);
    std::vector<Complex> er_field(Nx * Ny, er_val); // uniform field of size Nx * Ny
    er_field.insert(er_field.end(), squared_field.begin(), squared_field.end()); // insert additional layer --> not so nice, preallocate and slice
    std::cout << "er_field " << er_field.size() << '\n';
    std::vector<Complex> er{er_field};
    std::cout << "er " << er.size() << '\n';
    Complex ur_val(2.0, 0.0);
    std::vector<Complex> ur_field(Nx * Ny, ur_val); // uniform field of size Nx * Ny
    ur_field.insert(ur_field.end(), squared_field.begin(), squared_field.end());
    std::vector<Complex> ur{ur_field};
    //std::vector<Real> thickness{1.0, 1.0}; // remember brace initialization
    //int num_layers = 2;
    std::vector<Real> thickness{1.0}; // remember brace initialization
    int num_layers = 1;
    Real Lx = 1.0;
    Real Ly = 1.0;
    Device device(Nx, Ny, num_layers, Lx, Ly, er, ur, thickness, Nx_harmonics, Ny_harmonics);
    //std::cout << device.t.at(0) << '\n';
    //std::cout << device.t.at(1) << '\n';

    //std::cout << "Device erc" <<'\n';
    //std::cout << device.erc.at(0) << '\n';
    //std::cout << device.erc.at(0).rows() << " " << device.erc.at(0).cols() << '\n';
    Source source(2, M_PI/4, M_PI/4, 1, 1);
    //Source source(0.5, 0.0, 0.0, 1, 1);
    Complex er_ref(2.0, 0.0);
    Complex ur_ref(1.0, 0.0);
    //Complex er_ref(6.0, -0.13);
    //Complex ur_ref(3.5, -0.15);
    Complex er_trn(9.0, 0.0);
    Complex ur_trn(1.0, 0.0);
    RCWAParams params(Nx_harmonics, Ny_harmonics, er_ref, ur_ref, er_trn, ur_trn);
    int P = 2 * params.Nx_harmonics + 1;
    int Q = 2 * params.Ny_harmonics + 1;
    int PQ = P * Q;
    //std::cout << params.er_ref << '\n';
    std::vector<Complex> k_inc(3, Complex(0.0, 0.0));
    Vector Kx, Ky, Kz_ref, Kz_trn; ////
    Matrix W0(2*PQ, 2*PQ);
    Matrix V0(2*PQ, 2*PQ);
    Matrix W_ref(2*PQ, 2*PQ);
    Matrix W_trn(2*PQ, 2*PQ);


    // Build ConvMats
    for (int layer = 0; layer < device.num_layers; ++layer)
    {
        std::cout << "Layer " << layer << '\n';
        device.erc.at(layer) = ConvMat(device.er, layer, device.Nx, device.Ny, params.Nx_harmonics, params.Ny_harmonics); // perhaps pass as const ref params directly
        device.urc.at(layer) = ConvMat(device.ur, layer, device.Nx, device.Ny, params.Nx_harmonics, params.Ny_harmonics);
        //std::cout << device.erc.at(layer) << '\n';
        std::cout << "Size: " << device.erc.at(layer).rows() << " x " << device.erc.at(layer).cols() << '\n';
    }
    ComputeWaveVectors(device, source, params, k_inc, Kx, Ky, Kz_ref, Kz_trn);
    GapMedium(Kx, Ky, W0, V0);
    ScatteringMatrix S_device = SMatrixInit(params.Nx_harmonics, params.Ny_harmonics); // Initialize S_device S-Matrix
    ScatteringMatrix S_layer = ScatteringMatrix(Matrix::Zero(2*PQ, 2*PQ), Matrix::Zero(2*PQ, 2*PQ), Matrix::Zero(2*PQ, 2*PQ), Matrix::Zero(2*PQ, 2*PQ));
    for (int layer = 0; layer < device.num_layers; ++layer)
    {
        std::cout << "Layer " << layer << '\n';
        //std::cout << device.erc.at(layer) << '\n';
        //std::cout << device.erc.at(layer) << '\n';
        //std::cout << device.urc.at(layer) << '\n';
        S_layer = SMatrixLayer(layer, device, source, params, Kx, Ky, W0, V0);
        S_device = RedhefferProduct(S_device, S_layer);
        std::cout << S_device.S11 << '\n';
        std::cout << '\n';
        std::cout << S_device.S12 << '\n';
        std::cout << '\n';
        std::cout << S_device.S21 << '\n';
        std::cout << '\n';
        std::cout << S_device.S22 << '\n';
    }

    ScatteringMatrix S_ref = SMatrixReflection(params, Kx, Ky, Kz_ref, W0, V0, W_ref);
    ScatteringMatrix S_trn = SMatrixTransmission(params, Kx, Ky, Kz_trn, W0, V0, W_trn);

    ScatteringMatrix S_global = RedhefferProduct(S_ref, S_device);
    S_global = RedhefferProduct(S_global, S_trn);

    std::cout << S_global.S11 << '\n';
    std::cout << '\n';
    std::cout << S_global.S12 << '\n';
    std::cout << '\n';
    std::cout << S_global.S21 << '\n';
    std::cout << '\n';
    std::cout << S_global.S22 << '\n';

    Vector csrc(2*PQ);
    ComputeSourceModeCoeff(source, params, k_inc, W_ref, csrc);
    Vector r = Vector::Zero(3*PQ);
    ComputeReflectedField(params, S_global, csrc, Kx, Ky, Kz_ref, W_ref, r);
    std::cout << r << '\n';
    Vector t = Vector::Zero(3*PQ);
    ComputeTransmittedField(params, S_global, csrc, Kx, Ky, Kz_trn, W_trn, t);
    std::cout << '\n';
    std::cout << t << '\n';

    Matrix C = ConvMat(er_field, 0, Nx, Ny, Nx_harmonics, Ny_harmonics);
    //std::cout << C << '\n';
    //MatrixXcd C = MatrixXcd::Zero(Nx, Ny);
    //std::cout << "Size of C: " << C.rows() << " x " << C.cols() << std::endl;
    ScatteringMatrix A(C, C, C, C);
    ScatteringMatrix B(C, C, C, C);
    ScatteringMatrix res = RedhefferProduct(A, B);
    /*
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
    */    
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


