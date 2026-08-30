# rcwa-cuda

GPU-accelerated Rigorous Coupled-Wave Analysis (RCWA) for semiconductor metrology simulation.

## Project status

The current repository contains the completed sequential C++ baseline for the RCWA solver. The code is implemented in native C++ and it is suitable as the reference CPU implementation before adding CUDA acceleration and further GPU-specific optimizations.

The long-term goal of this project is to extend this baseline with GPU-accelerated kernels for the numerically intensive parts of the RCWA workflow.

## What is included

- RCWA 3D source code in C++, CPU baseline implementation
- Build configuration for the CPU baseline

## Dependencies

This project depends on:

- Eigen3 for matrix and vector algebra
- BLAS for optimized dense linear algebra kernels
- LAPACK for advanced linear algebra operations
- LAPACKE for the C interface to LAPACK
- OpenMP for shared-memory parallel execution in the CPU baseline
- CMake and a C++17 compiler

## Build

From the project root:

```bash
cmake -S rcwa3d -B rcwa3d/build
cmake --build rcwa3d/build --parallel
```

Then run the solver:

```bash
./rcwa3d/build/rcwa_main
```

## Notes

The build configuration enables Eigen to use optimized BLAS/LAPACK backends and OpenMP parallelization, which helps maintain performance while keeping the baseline implementation portable and easy to extend.


