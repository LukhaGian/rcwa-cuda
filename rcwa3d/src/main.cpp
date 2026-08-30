#include "rcwa.hpp"
#include <iostream>


int main()
{
    int Nx = 2048; // make sure it's always even
    int Ny = 2048;

    int Nx_harmonics = 5;
    int Ny_harmonics = 5;


    Real Lx = 1.0; // unit cell dimension along X axis
    Real Ly = 1.0; // unit cell dimension along Y axis

    // Build the device layers
    std::vector<std::vector<Complex>> er_layers = 
        {
            layer::triangle_pattern(Nx, Ny, Lx, Ly),
            layer::x_pattern(Nx, Ny, Lx, Ly),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::square_pattern(Nx, Ny, Complex(2.0, 0.0), Complex(5.0, 0.0)),
            layer::y_pattern(Nx, Ny, Lx, Ly),
            layer::uniform(Nx, Ny, Complex(4.0, 0.0)),
            layer::square_pattern(Nx, Ny, Complex(7.0, 0.0), Complex(1.0, 0.0))

        };
    std::vector<std::vector<Complex>> ur_layers = 
        {
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            layer::uniform(Nx, Ny, Complex(1.0, 0.0))

        };
    std::vector<Complex> er = layer::stack(er_layers);
    std::vector<Complex> ur = layer::stack(ur_layers);


    std::vector<Real> thickness{0.5, 0.3, 0.4, 0.6, 0.1, 0.25};

    // Build the device
    Device device(Nx, Ny, thickness.size(), Lx, Ly, er, ur, thickness, Nx_harmonics, Ny_harmonics);

    // Build the source
    Real lambda0 = 2.0;
    Real theta = M_PI / 6.0;
    Real phi = M_PI / 12.0;
    Real pte = 0.0; // TE pol
    Real ptm = 1.0; // TM pol
    Source source(lambda0, theta, phi, pte, ptm);

    // Build reflection and transmission regions
    Complex er_ref(2.0, 0.0);
    Complex ur_ref(1.0, 0.0);
    Complex er_trn(9.0, 0.0);
    Complex ur_trn(1.0, 0.0);
    RCWAParams params(Nx_harmonics, Ny_harmonics, er_ref, ur_ref, er_trn, ur_trn);

    // Declare Matrices and Vectors
    int P = 2 * params.Nx_harmonics + 1;
    int Q = 2 * params.Ny_harmonics + 1;
    int PQ = P * Q;

    std::vector<Complex> k_inc(3, Complex(0.0, 0.0));
    Vector Kx = Vector::Zero(PQ);
    Vector Ky = Vector::Zero(PQ);
    Vector Kz_ref = Vector::Zero(PQ);
    Vector Kz_trn = Vector::Zero(PQ);
    Matrix W0 = Matrix::Zero(2*PQ, 2*PQ); // Some matrices are very sparse...
    Matrix V0 = Matrix::Zero(2*PQ, 2*PQ);
    Matrix W_ref = Matrix::Zero(2*PQ, 2*PQ);
    Matrix W_trn = Matrix::Zero(2*PQ, 2*PQ);

    // Build ConvMats
    for (int layer = 0; layer < device.num_layers; ++layer)
    {
        std::cout << "Convolution for Layer " << layer << '\n';
        device.erc.at(layer) = ConvMat(device.er, layer, device.Nx, device.Ny, params.Nx_harmonics, params.Ny_harmonics);
        device.urc.at(layer) = ConvMat(device.ur, layer, device.Nx, device.Ny, params.Nx_harmonics, params.Ny_harmonics);
    }

    ComputeWaveVectors(device, source, params, k_inc, Kx, Ky, Kz_ref, Kz_trn);
    GapMedium(Kx, Ky, W0, V0);

    ScatteringMatrix S_device = SMatrixInit(params.Nx_harmonics, params.Ny_harmonics); // Initialize S_device S-Matrix
    ScatteringMatrix S_layer = ScatteringMatrix(Matrix::Zero(2*PQ, 2*PQ), Matrix::Zero(2*PQ, 2*PQ), Matrix::Zero(2*PQ, 2*PQ), Matrix::Zero(2*PQ, 2*PQ));
    
    for (int layer = 0; layer < device.num_layers; ++layer)
    {
        std::cout << "Layer " << layer << '\n';
        S_layer = SMatrixLayer(layer, device, source, params, Kx, Ky, W0, V0);
        S_device = RedhefferProduct(S_device, S_layer);  
    }
    std::cout << "Reflection" << '\n';
    ScatteringMatrix S_ref = SMatrixReflection(params, Kx, Ky, Kz_ref, W0, V0, W_ref);
    std::cout << "Transmission" << '\n';
    ScatteringMatrix S_trn = SMatrixTransmission(params, Kx, Ky, Kz_trn, W0, V0, W_trn);

    ScatteringMatrix S_global = RedhefferProduct(S_ref, S_device);
    S_global = RedhefferProduct(S_global, S_trn);

    Vector csrc(2*PQ);
    ComputeSourceModeCoeff(source, params, k_inc, W_ref, csrc);
    Vector r = Vector::Zero(3*PQ);
    ComputeReflectedField(params, S_global, csrc, Kx, Ky, Kz_ref, W_ref, r);
    Vector t = Vector::Zero(3*PQ);
    ComputeTransmittedField(params, S_global, csrc, Kx, Ky, Kz_trn, W_trn, t);

    Results results = ComputeDiffractionEfficiencies(params, r, t, k_inc, Kz_ref, Kz_trn);
    std::cout << "R + T = " << results.R_tot << " + " << results.T_tot << " = " << results.R_tot + results.T_tot<< '\n';
    

    return 0;
}


