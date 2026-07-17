cmake_minimum_required(VERSION 3.20)
project(RCWA VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

# Main executable
add_executable(rcwa_main
    src/main.cpp
)

target_sources(rcwa_main PRIVATE
    src/rcwa.cpp
)

target_include_directories(rcwa_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)


# Test executable
#add_executable(test_eigen
#    Testing/test_eigen.cpp
#)

#target_include_directories(test_eigen PRIVATE
#    ${EIGEN3_INCLUDE_DIR}
#)