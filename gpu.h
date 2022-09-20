#ifndef GPU_H
#define GPU_H

#include <cstdio>
#include <cstdlib>

#if defined(__CUDACC__)
using gpuError_t = cudaError_t;
typedef cudaError_t gpuError_t;
static const auto gpuSuccess = cudaSuccess;
static const auto gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
static const auto gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
#define gpuGetErrorString(err) cudaGetErrorString(err)
#define gpuMalloc(...) cudaMalloc(__VA_ARGS__)
#define gpuFree(...) cudaFree(__VA_ARGS__)
#define gpuMemcpy(...) cudaMemcpy(__VA_ARGS__)
#define gpuGetLastError() cudaGetLastError()
#define gpuDeviceSynchronize() cudaDeviceSynchronize()
#else
#include <hip/hip_runtime.h>
using gpuError_t = hipError_t;
static const auto gpuSuccess = hipSuccess;
static const auto gpuMemcpyHostToDevice = hipMemcpyHostToDevice;
static const auto gpuMemcpyDeviceToHost = hipMemcpyDeviceToHost;
#define gpuGetErrorString(err) hipGetErrorString(err)
#define gpuMalloc(...) hipMalloc(__VA_ARGS__)
#define gpuFree(...) hipFree(__VA_ARGS__)
#define gpuMemcpy(...) hipMemcpy(__VA_ARGS__)
#define gpuGetLastError() hipGetLastError()
#define gpuDeviceSynchronize() hipDeviceSynchronize()
#endif

#define GPU_CALL(x) do { gpuError_t err = x; if (( err ) != gpuSuccess ) { \
	printf ("Error \"%s\" at %s :%d \n" , gpuGetErrorString(err), \
		__FILE__ , __LINE__ ) ; exit(-1); \
}} while (0)


#endif // GPU_H

