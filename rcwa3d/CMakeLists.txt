# RCWA (Rigorous Coupled-Wave Analysis) 3D Solver
# Build config for numerical computing with OpenMP parallelization
cmake_minimum_required(VERSION 3.20)
project(RCWA VERSION 1.0 LANGUAGES CXX)

# C++ config
set(CMAKE_CXX_STANDARD 17) # Use C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Enforce C++17 requirement

# Dependency discovery
find_package(Eigen3 REQUIRED) # Linear algebra library
find_package(BLAS REQUIRED) # Basic Linear Algebra Subroutines (optimized kernels)
find_package(LAPACK REQUIRED) # Linear Algebra Package (advanced ops)
find_package(OpenMP REQUIRED) # Multi-threaded parallelization support

# Additional library search for LAPACKE (C interface to LAPACK)
find_library(LAPACKE_LIBRARY NAMES lapacke REQUIRED)

# RCWA Main Executable Target
add_executable(rcwa_main src/main.cpp)

# Add additional source files to the executable
target_sources(rcwa_main PRIVATE
    src/rcwa.cpp # Core RCWA algorithm implementation
)

# Specify include directories for compilation
target_include_directories(rcwa_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include # Project-specific headers
    ${EIGEN3_INCLUDE_DIR} # Eigen library headers
)

# Compiler optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(rcwa_main PRIVATE
        -O2 # Moderate optimization
        -march=native # Enable CPU-specific SIMD instructions
    )
else()
    target_compile_options(rcwa_main PRIVATE -O2) # Fallback for non-GCC/Clang compilers
endif()

# Enable Eigen library integration with optimized BLAS and LAPACK backends:
# this allows Eigen to delegate large matrix operations to optimized libraries.
target_compile_definitions(rcwa_main PRIVATE
    EIGEN_USE_BLAS # Use external BLAS for matrix op
    EIGEN_USE_LAPACKE # Use LAPACKE for advanced linear algebra
    EIGEN_USE_OPENMP # Use OpenMP for parallelization in Eigen
)

# Link against performance-critical libraries
target_link_libraries(rcwa_main PRIVATE
    ${BLAS_LIBRARIES}  # BLAS
    ${LAPACK_LIBRARIES} # LAPACK
    ${LAPACKE_LIBRARY} # LAPACKE
    OpenMP::OpenMP_CXX # OpenMP runtime for parallel computation

)

# Test executable
#add_executable(test_eigen
#    Testing/test_eigen.cpp
#)

#target_include_directories(test_eigen PRIVATE
#    ${EIGEN3_INCLUDE_DIR}
#)