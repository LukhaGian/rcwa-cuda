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

    int Nx_harmonics = 3;
    int Ny_harmonics = 3;


    // Build the device
    Real Lx = 1.75;
    Real Ly = 1.50;

    std::vector<Complex> er_field(Ny * Nx, Complex(0.0, 0.0));
    std::vector<Complex> ur_field(Ny * Nx, Complex(0.0, 0.0));

    for (int iy = 0; iy < Ny; ++iy)
    {
        double y = (iy + 0.5) * Ly / Ny;  // physical y coordinate
        for (int ix = 0; ix < Nx; ++ix)
            {
                double x = (ix + 0.5) * Lx / Nx;  // physical x coordinate
                
                // Triangle centred at (Lx/2, Ly/2) with base w = 0.8*Ly
                double w = 0.8 * Ly;
                double xc = Lx / 2.0;
                double yc = Ly / 2.0;
                
                // Triangle vertices (centred)
                double y_rel = y - (yc - w/2.0);  // relative to triangle base
                double half_base = (w/2.0) * (y_rel/w);  // apex at top, base at bottom                
                bool inside_triangle = (y_rel >= 0) && (y_rel <= w) && (std::abs(x - xc) <= half_base);
                
                er_field[0 * Ny * Nx + iy * Nx + ix] = inside_triangle ? Complex(2.0, 0.0) : Complex(6.0, 0.0);
                ur_field[0 * Ny * Nx + iy * Nx + ix] = inside_triangle ? Complex(1.0, 0.0) : Complex(1.0, 0.0);
            }
    }

    using RowMajorMatrixXdMap = Eigen::Map<const Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    //                                  raw pointer of data + move it to the specific layer, then map it to a row-major matrix of size Ny x Nx
    RowMajorMatrixXdMap realRowMajorMap(er_field.data(), Ny, Nx);
    // cast the real row-major matrix to a complex column-major matrix of size Ny x Nx (default Eigen matrix is column-major)
    //Matrix F = realRowMajorMap.cast<Complex>();
    Matrix F = realRowMajorMap; // no need to cast, data is already Complex

    //std::cout << F << '\n';


    std::vector<Complex> uniform_er_field(Ny * Nx, Complex(6.0, 0.0));
    std::vector<Complex> uniform_ur_field(Ny * Nx, Complex(1.0, 0.0));
    std::vector<Real> thickness{0.5, 0.3}; // remember brace initialization
    
    er_field.insert(er_field.end(), uniform_er_field.begin(), uniform_er_field.end());
    std::vector<Complex> er{er_field};
    ur_field.insert(ur_field.end(), uniform_ur_field.begin(), uniform_ur_field.end());
    std::vector<Complex> ur{ur_field};
    Device device(Nx, Ny, thickness.size(), Lx, Ly, er, ur, thickness, Nx_harmonics, Ny_harmonics);

    // Build the source
    Real lambda0 = 2.0;
    Real theta = 0.0;
    Real phi = 0.0;
    Real pte = 0.0;
    Real ptm = 1.0; // TM pol
    //Source source(0.5, M_PI/4, M_PI/4, 0.5, 0.5);
    Source source(lambda0, theta, phi, pte, ptm);

    // Build reflection and transmission region
    Complex er_ref(2.0, 0.0);
    Complex ur_ref(1.0, 0.0);
    //Complex er_ref(6.0, -0.13);
    //Complex ur_ref(3.5, -0.15);
    Complex er_trn(9.0, 0.0);
    Complex ur_trn(1.0, 0.0);
    RCWAParams params(Nx_harmonics, Ny_harmonics, er_ref, ur_ref, er_trn, ur_trn);

    // Declare Matrices and Vectors
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
        std::cout << "Convolution for Layer " << layer << '\n';
        device.erc.at(layer) = ConvMat(device.er, layer, device.Nx, device.Ny, params.Nx_harmonics, params.Ny_harmonics); // perhaps pass as const ref params directly
        device.urc.at(layer) = ConvMat(device.ur, layer, device.Nx, device.Ny, params.Nx_harmonics, params.Ny_harmonics);
        /*
        std::cout << device.erc.at(layer) << '\n';
        std::cout << "Size: " << device.erc.at(layer).rows() << " x " << device.erc.at(layer).cols() << '\n';
        std::cout << device.urc.at(layer) << '\n';
        std::cout << "Size: " << device.urc.at(layer).rows() << " x " << device.urc.at(layer).cols() << '\n';
        std::cout << '\n';
        */
    }

    ComputeWaveVectors(device, source, params, k_inc, Kx, Ky, Kz_ref, Kz_trn);
    GapMedium(Kx, Ky, W0, V0);

    ScatteringMatrix S_device = SMatrixInit(params.Nx_harmonics, params.Ny_harmonics); // Initialize S_device S-Matrix
    ScatteringMatrix S_layer = ScatteringMatrix(Matrix::Zero(2*PQ, 2*PQ), Matrix::Zero(2*PQ, 2*PQ), Matrix::Zero(2*PQ, 2*PQ), Matrix::Zero(2*PQ, 2*PQ));
    
    for (int layer = 0; layer < device.num_layers; ++layer)
    {
        std::cout << "Layer " << layer << '\n';
        //std::cout << device.erc.at(layer) << '\n';
        //std::cout << device.urc.at(layer) << '\n';
        S_layer = SMatrixLayer(layer, device, source, params, Kx, Ky, W0, V0);
      
        S_device = RedhefferProduct(S_device, S_layer);
        /*
        std::cout << S_device.S11 << '\n';
        std::cout << '\n';
        std::cout << S_device.S12 << '\n';
        std::cout << '\n';
        std::cout << S_device.S21 << '\n';
        std::cout << '\n';
        std::cout << S_device.S22 << '\n';
        */
        
    }
    
    ScatteringMatrix S_ref = SMatrixReflection(params, Kx, Ky, Kz_ref, W0, V0, W_ref);
    ScatteringMatrix S_trn = SMatrixTransmission(params, Kx, Ky, Kz_trn, W0, V0, W_trn);

    ScatteringMatrix S_global = RedhefferProduct(S_ref, S_device);
    S_global = RedhefferProduct(S_global, S_trn);

    /*
    std::cout << S_global.S11 << '\n';
    std::cout << '\n';
    std::cout << S_global.S12 << '\n';
    std::cout << '\n';
    std::cout << S_global.S21 << '\n';
    std::cout << '\n';
    std::cout << S_global.S22 << '\n';
    */

    Vector csrc(2*PQ);
    ComputeSourceModeCoeff(source, params, k_inc, W_ref, csrc);
    Vector r = Vector::Zero(3*PQ);
    ComputeReflectedField(params, S_global, csrc, Kx, Ky, Kz_ref, W_ref, r);
    //std::cout << r << '\n';
    Vector t = Vector::Zero(3*PQ);
    ComputeTransmittedField(params, S_global, csrc, Kx, Ky, Kz_trn, W_trn, t);
    std::cout << '\n';
    //std::cout << t << '\n';
    Results results = ComputeDiffractionEfficiencies(params, r, t, k_inc, Kz_ref, Kz_trn);
    std::cout << "R" << '\n';
    std::cout << results.R << '\n';
    std::cout << '\n';
    std::cout << "T" << '\n';
    std::cout << results.T << '\n';
    std::cout << results.R_tot << "+" << results.T_tot << '\n';
    std::cout << results.R_tot + results.T_tot << '\n';

    

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


