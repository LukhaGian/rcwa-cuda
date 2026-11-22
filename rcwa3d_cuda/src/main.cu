#include "rcwa_cuda.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <thread> // for std::thread
#include <functional> // for std::ref and std::cref, avoid expensive copies 
#include <chrono> // timing

int main()
{

    // Build all the sources that we will simulate
    int num_wavelengths = 100;  // # of sources to simulate

    std::vector<Real> lambda0_v(num_wavelengths);
    std::vector<Real> k0_v(num_wavelengths);
    Real lambda_start = 1.0;
    Real lambda_end = 2.0;
    // If num_wavelengths is 1, just set it to the lambda_start value to avoid division by zero
    if (num_wavelengths == 1)
    {
        lambda0_v[0] = lambda_start;
        k0_v[0] = 2 * M_PI / lambda0_v[0];
    }
    else 
    {
        Real step = (lambda_end - lambda_start) / (num_wavelengths - 1);
        for (int i = 0; i < num_wavelengths; ++i) 
        {
            lambda0_v[i] = lambda_start + i * step;
            k0_v[i] = 2 * M_PI / lambda0_v[i];
        }
    }

    Real theta = M_PI / 6.0;
    std::vector<Real> theta_v(num_wavelengths, theta); // All thetas are the same for now
    Real phi = M_PI / 12.0;
    std::vector<Real> phi_v(num_wavelengths, phi); // All phis are the same for now
    Real pte = 0.0; // TE pol
    std::vector<Real> pte_v(num_wavelengths, pte); // All ptes are the same for now
    Real ptm = 1.0; // TM pol
    std::vector<Real> ptm_v(num_wavelengths, ptm); // All ptms are the same for now
    Source source(lambda0_v, k0_v, theta_v, phi_v, pte_v, ptm_v);



    // spatial grid for the layers
    int Nx = 2048; // make sure it's always even
    int Ny = 2048;

    int Nx_harmonics = 1;
    int Ny_harmonics = 1;

    int P = 2 * Nx_harmonics + 1; // total number of harmonics along x axis
    int Q = 2 * Ny_harmonics + 1; // total number of harmonics along y axis
    int PQ = P * Q;

    Real Lx = 1.0; // unit cell dimension along x axis
    Real Ly = 1.0; // unit cell dimension along y axis

    Real dx = Lx/Nx;
    Real dy = Ly/Ny;

    std::vector<Complex> values{Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(5.0, 0.0), Complex(6.0, 0.0), Complex(8.0, 0.0)};
    ValueTable table{static_cast<int>(values.size()), num_wavelengths, {}};
    table.values.reserve(values.size() * num_wavelengths);
    for (int i = 0; i < num_wavelengths; ++i) // same row for every wavelength, for now
        table.values.insert(table.values.end(), values.begin(), values.end());  

    // Build the device layers
    ShapeDesc er_0{SHAPE_TRIANGLE, 0.7, -1.0, -1.0, dx, dy, 1, 3};
    ShapeDesc er_1{SHAPE_X, -1.0, -1.0, -1.0, dx, dy, 1, 3};
    ShapeDesc er_2{SHAPE_UNIFORM, -1.0, -1.0, -1.0, dx, dy, 0, 0};
    ShapeDesc er_3{SHAPE_TRIANGLE, 0.7, -1.0, -1.0, dx, dy, 1, 3};
   
    std::vector<std::vector<Complex>> er_layers = 
    {
            layer::build(er_0, Nx, Ny, Lx, Ly, table.values),
            layer::build(er_1, Nx, Ny, Lx, Ly, table.values),
            layer::build(er_2, Nx, Ny, Lx, Ly, table.values),
            layer::build(er_3, Nx, Ny, Lx, Ly, table.values)
    };
    ShapeDesc ur_0{SHAPE_TRIANGLE_SHIFTED, 0.7, 0.35, 0.35, dx, dy, 0, 2};
    ShapeDesc ur_1{SHAPE_UNIFORM, -1.0, -1.0, -1.0, dx, dy, 0, 0};
    ShapeDesc ur_2{SHAPE_UNIFORM, -1.0, -1.0, -1.0, dx, dy, 1, 1};
    ShapeDesc ur_3{SHAPE_SQUARE, std::min(Nx, Ny)/4.0, -1.0, -1.0, dx, dy, 1, 4};
    // Build device descriptors
    std::vector<ShapeDesc> er_desc{er_0, er_1, er_2, er_3};
    std::vector<ShapeDesc> ur_desc{ur_0, ur_1, ur_2, ur_3};


    std::vector<std::vector<Complex>> ur_layers = 
    {
            layer::build(ur_0, Nx, Ny, Lx, Ly, table.values),
            layer::build(ur_1, Nx, Ny, Lx, Ly, table.values),
            layer::build(ur_2, Nx, Ny, Lx, Ly, table.values),
            layer::build(ur_3, Nx, Ny, Lx, Ly, table.values) 
    };
    
    std::vector<Complex> er = layer::stack(er_layers);
    std::vector<Complex> ur = layer::stack(ur_layers);

    std::vector<Real> thickness{1.5, 2.0, 1.0, 2.0};

    // Build the device --> for now materials are not wavelength dependent
    Device device(Nx, Ny, thickness.size(), Lx, Ly, er_desc, ur_desc, thickness, Nx_harmonics, Ny_harmonics);



    // Build reflection and transmission regions --> for now the materials are non wavelength dependent, extension to wavelength dependent materials is possible
    Complex er_ref(2.0, 0.0);
    Complex ur_ref(1.0, 0.0);
    Complex er_trn(9.0, 0.0);
    Complex ur_trn(1.0, 0.0);
    RCWAParams params(Nx_harmonics, Ny_harmonics, er_ref, ur_ref, er_trn, ur_trn);

  


    // CUDA

    int num_gpus = 0; // init variable
    CHECK(cudaGetDeviceCount(&num_gpus)); // get the number of available gpus
    std::cout << "no. of gpus: " << num_gpus << '\n';
    for (int g = 0; g < num_gpus; ++g)
    {
        CHECK(cudaSetDevice(g));
        CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)); // waiting threads sleep instead of spinning
    }

    bool debug = true;
    std::atomic<int> next_wavelength{0};


    const int num_workers = num_gpus; // later: workers_per_gpu * num_gpus
    // create a vector of workers
    std::vector<StreamWorker> workers(num_workers);
    // Initialize the workers
    for (int i = 0; i < num_workers; ++i)
        InitWorker(workers[i], i % num_gpus, device, table, params, debug && i == 0); // active debug params only for 0th worker 

    // timing
    const auto wall_start = std::chrono::steady_clock::now();
    // vector of threads
    std::vector<std::thread> threads;
    threads.reserve(num_workers); // reserve space for all workers
    // Run each worker
    for (int k = 0; k < num_workers; ++k)
        threads.emplace_back(RunWorker, std::ref(workers[k]), std::cref(device), std::cref(source), std::cref(params), std::ref(next_wavelength));
    for (auto& t : threads)
        t.join(); // it locks the calling thread until the worker thread finishes execution
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();



    // Single worker
    /*
    std::atomic<int> next_wavelength{0};
    StreamWorker worker;
    bool debug = true;
    InitWorker(worker, 0, device, params, debug);
    
    // RunWorker in std::thread. 
    // Here we use std::ref since it allows us to pass by reference to std::thread (by default it copies the data even if the thread function uses references). 
    // Similarly, std::cref passes its argument by const reference.
    // When you pass arguments to a std::thread constructor, it internally copies or moves 
    // all arguments into the new thread's memory space, even if your target function takes a reference.
    const auto wall_start = std::chrono::steady_clock::now();
    std::thread t(RunWorker, std::ref(worker), std::cref(device), std::cref(source), std::cref(params), std::ref(next_wavelength));
    t.join(); // it locks the calling thread until the worker thread finishes execution
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    // Convolution results
    const auto* erc = reinterpret_cast<const Complex*>(worker.h_erc); // raw pointer
    const auto* urc = reinterpret_cast<const Complex*>(worker.h_urc);
    */



    //////////////////////////////////////////////////
    // TESTING
    //////////////////////////////////////////////////
    // CHECK WILL FAIL HERE
    // void* p = nullptr;
    // CHECK(cudaMalloc(&p, size_t(1) << 50)); // 1 PB



    // Print no. of wavelengths computed by each worker
    for (int i = 0; i < num_workers; ++i)
        std::cout << "worker no. " << i << ", num_solved: " << workers[i].num_solved << '\n';

    // convMat, Toeplitz, multiple workers -> only 0th worker has debug enabled
    if (workers[0].has_debug)
    {
        // Convolution results for 0th worker
        const auto* erc = reinterpret_cast<const Complex*>(workers[0].h_erc); // raw pointer
        const auto* urc = reinterpret_cast<const Complex*>(workers[0].h_urc);
        int check_layer = device.num_layers;
        size_t lower_bound = (check_layer-1) * static_cast<size_t>(PQ) * PQ;
        size_t upper_bound = (check_layer) * static_cast<size_t>(PQ) * PQ;
        const size_t convmat_elements = static_cast<size_t>(PQ) * PQ * device.num_layers;
        std::cout << "Size of erc " << convmat_elements << '\n';
        for (size_t i = lower_bound; i < upper_bound; ++i) 
            std::cout << erc[i] << '\n';
        std::cout << "----------------------------------------" << '\n';
        std::cout << "Size of urc " << convmat_elements << '\n'; 
        for (size_t i = lower_bound; i < upper_bound; ++i) 
            std::cout << urc[i] << '\n';
    }

    std::cout << "wall_s: " << wall_s << '\n';

    // Destroy workers
    for (auto& w : workers)
        DestroyWorker(w);

    return 0;
}