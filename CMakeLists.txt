cmake_minimum_required(VERSION 3.19)

project(bilinear CXX CUDA)

find_package(OpenMP REQUIRED)

add_library(easybmp EasyBMP/EasyBMP.cpp)
target_include_directories(easybmp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/EasyBMP)

add_executable(bilinear_cpu bilinear.cpp)
target_compile_features(bilinear_cpu PUBLIC cxx_std_11)
target_link_libraries(bilinear_cpu easybmp OpenMP::OpenMP_CXX)

add_executable(bilinear_gpu_unoptimized unoptimized/bilinear.cu)
target_compile_features(bilinear_gpu_unoptimized PUBLIC cxx_std_11)
target_link_libraries(bilinear_gpu_unoptimized easybmp)
set_target_properties(bilinear_gpu_unoptimized PROPERTIES CUDA_ARCHITECTURES "50;86")

