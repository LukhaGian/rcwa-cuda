cmake_minimum_required(VERSION 3.18)
project(CudaTest LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target Kaggle's Tesla T4 GPU (sm_75)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find the CUDA toolkit (ensure it includes the cufft component) ////////////////// To Be Checked
find_package(CUDAToolkit REQUIRED COMPONENTS cufft)

# Standalone CUDA executable
add_executable(rcwa_cu_main src/main.cu)
# Tell CMake to enable separable compilation for the CUDA executable and then perform a CUDA device-link step
set_target_properties(rcwa_cu_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add additional source files to the executable
target_sources(rcwa_cu_main PRIVATE
    src/rcwa_cuda.cu # Core RCWA algorithm implementation
)

# Specify include directories for compilation
target_include_directories(rcwa_cu_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include # Project-specific headers
)

# Add the cuFFT library linking step
target_link_libraries(rcwa_cu_main PRIVATE CUDA::cufft) # /////////////////// To Be Checked

# Optimization flags for nvcc
target_compile_options(rcwa_cu_main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)