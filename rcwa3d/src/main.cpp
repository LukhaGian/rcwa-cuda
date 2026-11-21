#include "rcwa.hpp"
#include <iostream>
#include <chrono>


int main()
{
    #if defined(EIGEN_USE_BLAS)
        std::cout << "Eigen is configured to use external BLAS.\n";
    #else
        std::cout << "Eigen is NOT using external BLAS.\n";
    #endif

    #if defined(EIGEN_USE_LAPACKE)
        std::cout << "Eigen is configured to use LAPACKE.\n";
    #else
        std::cout << "Eigen is NOT using LAPACKE.\n";
    #endif

    int Nx = 2048; // make sure it's always even
    int Ny = 2048;

    int Nx_harmonics = 10;
    int Ny_harmonics = 10;


    Real Lx = 1.0; // unit cell dimension along X axis
    Real Ly = 1.0; // unit cell dimension along Y axis

    Real dx = Lx/Nx;
    Real dy = Ly/Ny;

    std::vector<Complex> values{Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(5.0, 0.0), Complex(6.0, 0.0), Complex(8.0, 0.0)};

    // Build the device layers
    ShapeDesc er_0{SHAPE_TRIANGLE, 0.7, -1.0, -1.0, dx, dy, 1, 3};
    ShapeDesc er_1{SHAPE_X, -1.0, -1.0, -1.0, dx, dy, 1, 3};
    ShapeDesc er_2{SHAPE_UNIFORM, -1.0, -1.0, -1.0, dx, dy, 0, 0};
    ShapeDesc er_3{SHAPE_TRIANGLE, 0.7, -1.0, -1.0, dx, dy, 1, 3};
   
    std::vector<std::vector<Complex>> er_layers = 
    {
            layer::build(er_0, Nx, Ny, Lx, Ly, values),
            layer::build(er_1, Nx, Ny, Lx, Ly, values),
            layer::build(er_2, Nx, Ny, Lx, Ly, values),
            layer::build(er_3, Nx, Ny, Lx, Ly, values)
            
            //legacy_layer::triangle_pattern(Nx, Ny, Lx, Ly, Complex(2.0, 0.0), Complex(6.0, 0.0)),
            //legacy_layer::x_pattern(Nx, Ny, Lx, Ly, Complex(2.0, 0.0), Complex(6.0, 0.0)),
            //legacy_layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            //legacy_layer::triangle_pattern(Nx, Ny, Lx, Ly, Complex(2.0, 0.0), Complex(6.0, 0.0))
            
    };
    ShapeDesc ur_0{SHAPE_TRIANGLE_SHIFTED, 0.7, 0.35, 0.35, dx, dy, 0, 2};
    ShapeDesc ur_1{SHAPE_UNIFORM, -1.0, -1.0, -1.0, dx, dy, 0, 0};
    ShapeDesc ur_2{SHAPE_UNIFORM, -1.0, -1.0, -1.0, dx, dy, 1, 1};
    ShapeDesc ur_3{SHAPE_SQUARE, std::min(Nx, Ny)/4.0, -1.0, -1.0, dx, dy, 1, 4};


    std::vector<std::vector<Complex>> ur_layers = 
    {
            layer::build(ur_0, Nx, Ny, Lx, Ly, values),
            layer::build(ur_1, Nx, Ny, Lx, Ly, values),
            layer::build(ur_2, Nx, Ny, Lx, Ly, values),
            layer::build(ur_3, Nx, Ny, Lx, Ly, values)
            
            //legacy_layer::triangle_pattern_shifted(Nx, Ny, Lx, Ly, 0.35, 0.35, Complex(1.0, 0.0), Complex(5.0, 0.0)),
            //legacy_layer::uniform(Nx, Ny, Complex(1.0, 0.0)),
            //legacy_layer::uniform(Nx, Ny, Complex(2.0, 0.0)),
            //legacy_layer::square_pattern(Nx, Ny, Complex(2.0, 0.0), Complex(8.0, 0.0))
            
    };
    std::vector<Complex> er = layer::stack(er_layers);
    std::vector<Complex> ur = layer::stack(ur_layers);

    std::vector<Real> thickness{1.5, 2.0, 1.0, 2.0};


    // Build the device
    Device device(Nx, Ny, thickness.size(), Lx, Ly, er, ur, thickness, Nx_harmonics, Ny_harmonics);

    // Build reflection and transmission regions
    Complex er_ref(2.0, 0.0);
    Complex ur_ref(1.0, 0.0);
    Complex er_trn(9.0, 0.0);
    Complex ur_trn(1.0, 0.0);
    RCWAParams params(Nx_harmonics, Ny_harmonics, er_ref, ur_ref, er_trn, ur_trn);


    // Build ConvMats
    for (int layer = 0; layer < device.num_layers; ++layer)
    {
        std::cout << "Convolution for Layer " << layer << '\n';
        device.erc.at(layer) = ConvMat(device.er, layer, device.Nx, device.Ny, params.Nx_harmonics, params.Ny_harmonics);
        device.urc.at(layer) = ConvMat(device.ur, layer, device.Nx, device.Ny, params.Nx_harmonics, params.Ny_harmonics);
        if (layer == -1)
        {
            std::cout << device.erc.at(layer) << '\n';
            std::cout << '\n';
            std::cout << device.urc.at(layer) << '\n';
        }
    }

    // Wavelength sweep parameters
    int num_wavelengths = 8;

    std::vector<Real> lambda0_v(num_wavelengths, -1.0);
    Real lambda_start = 2.0;
    Real lambda_end = 3.0;
    // If num_wavelength is 1, just set it to the lambda_start value to avoid division by zero
    if (num_wavelengths == 1)
        lambda0_v[0] = lambda_start;
    else 
    {
        Real step = (lambda_end - lambda_start) / (num_wavelengths - 1);
        for (int i = 0; i < num_wavelengths; ++i) 
            lambda0_v[i] = lambda_start + i * step;
    }

    // Remaining source parameters
    Real theta = M_PI / 6.0;
    Real phi = M_PI / 12.0;
    Real pte = 0.0; // TE pol
    Real ptm = 1.0; // TM pol
    
    std::vector<Real> R_tot(num_wavelengths, -1.0);
    std::vector<Real> T_tot(num_wavelengths, -1.0);

    const auto wall_start = std::chrono::steady_clock::now();
    /*
    Parallelize the wavelength sweep using OpenMP:
    -) parallel: starts a team of threads: no. of threads = OMP_NUM_THREADS
    -) schedule(dynamic): each thread gets an iteration at a time and asks for another one
        when it finishes. dynamic scheduling is useful when the iterations have varying execution times,
        which is the case here since different wavelengths may take different amounts of time to compute 
        (small overhead compared to static scheduling).
    */
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_wavelengths; ++i)
    {
        // Build the source
        const Source source(lambda0_v[i], theta, phi, pte, ptm);
        const Results results = RCWASolveWavelength(device, source, params);
        R_tot[i] = results.R_tot;
        T_tot[i] = results.T_tot;
    }
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    
    for (int i = 0; i < num_wavelengths; ++i)
    {
        std::cout << "lambda = " << lambda0_v[i] << ": R + T = " << R_tot[i] << " + " << T_tot[i] << " = " << R_tot[i] + T_tot[i] << '\n';
    }
    std::cout << "Wall time: " << wall_s << " s" << '\n';

    return 0;
}


