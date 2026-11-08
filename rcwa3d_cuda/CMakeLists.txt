cmake_minimum_required(VERSION 3.20)
project(RCWA_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target SM arch for NVIDIA GPUs.
set(CMAKE_CUDA_ARCHITECTURES 75; 80; 90) # A100 (sm_80) and H100 (sm_90)

# Find the CUDA toolkit, which provides the CUDA::cufft, CUDA::cusolver and CUDA::cublas targets
find_package(CUDAToolkit REQUIRED)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared/include # Helpers shared with the other build
)

# Add the cuFFT library linking step
target_link_libraries(rcwa_cu_main PRIVATE CUDA::cufft CUDA::cusolver CUDA::cublas)

# Optimization flags for nvcc
target_compile_options(rcwa_cu_main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Standalone cusolverDnXgeev benchmark
add_executable(geev_bench tests/geev_bench.cu)
target_include_directories(geev_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(geev_bench PRIVATE CUDA::cusolver CUDA::cublas CUDA::cudart)
target_compile_options(geev_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)