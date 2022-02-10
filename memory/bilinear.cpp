#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <vector>
#include "EasyBMP.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

extern unsigned char bilinear_cu[];
extern unsigned int bilinear_cu_len;

#define NVRTC_CALL(x)                                             \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_CALL(x)                                              \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDART_CALL(x)                                            \
  do {                                                            \
    cudaError_t err = x;                                          \
    if (( err ) != cudaSuccess ) {                                \
	  printf ("Error \"%s\" at %s :%d \n" ,                       \
	    cudaGetErrorString(err),                                  \
		__FILE__ , __LINE__ ) ; exit(-1);                         \
    }                                                             \
  } while (0)

using namespace std;

// Get the timer value.
static void get_time(volatile struct timespec* val)
{
	clock_gettime(CLOCK_REALTIME, (struct timespec*)val);
}

// Get the timer measured values difference.
static double get_time_diff(struct timespec* val1, struct timespec* val2)
{
	int64_t seconds = val2->tv_sec - val1->tv_sec;
	int64_t nanoseconds = val2->tv_nsec - val1->tv_nsec;
	if (val2->tv_nsec < val1->tv_nsec)
	{
		seconds--;
		nanoseconds = (1000000000 - val1->tv_nsec) + val2->tv_nsec;
	}
	
	return (double)0.000000001 * nanoseconds + seconds;
}
 
int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cout << "Usage: " << argv[0] << " <filename>" << endl;
		return 0;
	}

	char* filename = argv[1];

	BMP AnImage;
	AnImage.ReadFromFile(filename);
	int width = AnImage.TellWidth();
	int height = AnImage.TellHeight();

	vector<RGBApixel> input(width * (height + 1) + 1);
	vector<RGBApixel> output(4 * width * height);
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			input[i + j * width] = AnImage.GetPixel(i, j);
	memset(&input[height * width], 0, (width + 1) * sizeof(RGBApixel));

	// Initialize CUDA driver.
	CUdevice cuDevice;
	CUcontext context;
	CUDA_CALL(cuInit(0));
	CUDA_CALL(cuDeviceGet(&cuDevice, 0));
	CUDA_CALL(cuCtxCreate(&context, 0, cuDevice));

	// Create an instance of nvrtcProgram with the SAXPY code string.
	nvrtcProgram prog;
	NVRTC_CALL(nvrtcCreateProgram(&prog, (char*)bilinear_cu, "bilinear", 0, NULL, NULL));

	// Compile the program for compute_30.
	const char *opts[] = { "--gpu-architecture=compute_30" };
	nvrtcResult compileResult = nvrtcCompileProgram(prog, sizeof(opts) / sizeof(char*), opts);

	// Obtain compilation log from the program.
	size_t logSize;
	NVRTC_CALL(nvrtcGetProgramLogSize(prog, &logSize));
	if (logSize > 1)
	{
		std::vector<char> log(logSize);
		NVRTC_CALL(nvrtcGetProgramLog(prog, &log[0]));
		std::cout << &log[0] << std::endl;
	}
	if (compileResult != NVRTC_SUCCESS)
		exit(1);

	// Obtain PTX from the program.
	size_t ptxSize;
	NVRTC_CALL(nvrtcGetPTXSize(prog, &ptxSize));
	vector<char> ptx;
	ptx.resize(ptxSize + 1);
	NVRTC_CALL(nvrtcGetPTX(prog, &ptx[0]));
	ptx[ptxSize] = '\0';

	// Destroy the program.
	NVRTC_CALL(nvrtcDestroyProgram(&prog));

	CUmodule module;
	CUDA_CALL(cuModuleLoadData(&module, &ptx[0]));
	CUfunction function;
	CUDA_CALL(cuModuleGetFunction(&function, module, "bilinear"));

	RGBApixel *dinput, *doutput;
	CUDART_CALL(cudaMalloc(&dinput, sizeof(RGBApixel) * input.size()));
	CUDART_CALL(cudaMalloc(&doutput, sizeof(RGBApixel) * output.size()));
	CUDART_CALL(cudaMemcpy(dinput, &input[0], sizeof(RGBApixel) * input.size(), cudaMemcpyHostToDevice));

	struct timespec start;
	get_time(&start);

	dim3 szblock(128, 1, 1);
	dim3 nblocks(2 * width / szblock.x, 2 * height, 1);
	if (2 * width % szblock.x) nblocks.x++;
	
	void *args[] = { &width, &height, &dinput, &doutput };
	CUDA_CALL(cuLaunchKernel(function, nblocks.x, nblocks.y, nblocks.z,
		szblock.x, szblock.y, szblock.z, 0, 0, args, 0));
	CUDART_CALL(cudaDeviceSynchronize());

	struct timespec finish;
	get_time(&finish);
	
	printf("GPU kernel time = %f sec\n", get_time_diff(&start, &finish));

	CUDART_CALL(cudaMemcpy(&output[0], doutput, sizeof(RGBApixel) * output.size(), cudaMemcpyDeviceToHost));

	AnImage.SetSize(2 * width, 2 * height);
	for (int i = 0; i < 2 * width; i++)
		for (int j = 0; j < 2 * height; j++)
			AnImage.SetPixel(i, j, output[i + j * 2 * width]);

	AnImage.WriteToFile("output_gpu.bmp");

	CUDART_CALL(cudaFree(dinput));
	CUDART_CALL(cudaFree(doutput));

	CUDA_CALL(cuModuleUnload(module));
	CUDA_CALL(cuCtxDestroy(context));
 
	return 0;
}

