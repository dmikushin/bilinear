cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(bilinear C CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_subdirectory(ThirdParty/res_embed EXCLUDE_FROM_ALL)

include(ResEmbed)

add_library(easybmp EasyBMP/EasyBMP.cpp)
target_include_directories(easybmp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/EasyBMP)

find_package(OpenCL REQUIRED)

add_executable(bilinear_opencl bilinear.cpp)
target_compile_features(bilinear_opencl PUBLIC cxx_std_11)
target_include_directories(bilinear_opencl PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(bilinear_opencl easybmp ${OpenCL_LIBRARIES})
res_embed(bilinear_opencl "bilinear_opencl" ${CMAKE_CURRENT_SOURCE_DIR}/opencl/bilinear.cl)

