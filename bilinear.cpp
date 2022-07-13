#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <vector>
#include "EasyBMP.h"
#include "res_embed.h"

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>
#include <CL/cl_ext.h>

extern unsigned char bilinear_cl[];
extern unsigned int bilinear_cl_len;

#define CL_CALL(x) do { cl_int err = x; if (( err ) != CL_SUCCESS ) { \
	printf ("Error \"%d\" at %s :%d \n" , err, \
		__FILE__ , __LINE__ ) ; exit(-1); \
}} while (0)

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

	cl_int status = CL_SUCCESS;

	// Initialize platform
	cl_uint numPlatforms = 0;
	CL_CALL(clGetPlatformIDs(0, NULL, &numPlatforms));
	cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
	CL_CALL(clGetPlatformIDs(numPlatforms, platforms,NULL));

	// Initialize device
	cl_uint numDevices = 0;
	CL_CALL(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices));
	if (numDevices == 0)
	{
		fprintf(stderr, "No OpenCL GPU devices detected, exiting...\n");
		exit(1);
	}
	cl_device_id* devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
	CL_CALL(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL));

	// Get device name, vendor and OpenCL version
	char name[1024];
#ifdef CL_DEVICE_BOARD_NAME_AMD
	if (clGetDeviceInfo(devices[0], CL_DEVICE_BOARD_NAME_AMD, sizeof(name) / sizeof(char), &name, NULL) != CL_SUCCESS)
#endif
	{
		CL_CALL(clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(name) / sizeof(char), &name, NULL));
	}
	char vendor[1024];
	CL_CALL(clGetDeviceInfo(devices[0], CL_DEVICE_VENDOR, sizeof(name) / sizeof(char), &vendor, NULL));
	char version[1024];
	CL_CALL(clGetDeviceInfo(devices[0], CL_DEVICE_OPENCL_C_VERSION, sizeof(name) / sizeof(char), &version, NULL));
	printf("Using OpenCL device \"%s\" produced by \"%s\" with %s\n", name, vendor, version);

	// Create context
	cl_context context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	// Create command queue
	cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
	CL_CALL(status);

	// Read and compile the program
	size_t size = 0;
	const char* source = res::embed::get("bilinear_opencl", &size);
	cl_program program = clCreateProgramWithSource(context, 1, &source, &size, &status);
	CL_CALL(status);
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	for (int i = 0; i < numDevices; i++)
	{
		size_t size = 0;
		CL_CALL(clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &size));
		vector<char> log(size);
		CL_CALL(clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, size, &log[0], NULL));
		printf("Building kernel for device #%d:\n%s", i, &log[0]); 
	}
	CL_CALL(status);

	// Create compiled kernel
	cl_kernel bilinear = clCreateKernel(program, "bilinear", &status);
	CL_CALL(status);

	cl_mem dinput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(RGBApixel) * input.size(), NULL, &status);
	CL_CALL(status);
	cl_mem doutput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(RGBApixel) * output.size(), NULL, &status);
	CL_CALL(status);
	CL_CALL(clEnqueueWriteBuffer(cmdQueue, dinput, CL_TRUE, 0, sizeof(RGBApixel) * input.size(), &input[0], 0, NULL, NULL));

	struct timespec start;
	get_time(&start);

	size_t szblock[2] = { 128, 1 };
	size_t a_width = 2 * width;
	size_t a_height = 2 * height;
	if (a_width % szblock[0])
		a_width += szblock[0] - a_width % szblock[0];
	size_t szproblem[2] = { a_width, a_height };
	int narg = 0;
	CL_CALL(clSetKernelArg(bilinear, narg++, sizeof(cl_int), &width));
	CL_CALL(clSetKernelArg(bilinear, narg++, sizeof(cl_int), &height));
	CL_CALL(clSetKernelArg(bilinear, narg++, sizeof(cl_mem), &dinput));
	CL_CALL(clSetKernelArg(bilinear, narg++, sizeof(cl_mem), &doutput));
	cl_event event;
	CL_CALL(clEnqueueNDRangeKernel(cmdQueue, bilinear, 2, NULL, szproblem, szblock, 0, NULL, &event));
	CL_CALL(clWaitForEvents(1, &event));

	struct timespec finish;
	get_time(&finish);
	
	printf("GPU kernel time = %f sec\n", get_time_diff(&start, &finish));

	CL_CALL(clEnqueueReadBuffer(cmdQueue, doutput, CL_TRUE, 0, sizeof(RGBApixel) * output.size(), &output[0], 0, NULL, NULL));

	AnImage.SetSize(2 * width, 2 * height);
	for (int i = 0; i < 2 * width; i++)
		for (int j = 0; j < 2 * height; j++)
			AnImage.SetPixel(i, j, output[i + j * 2 * width]);

	AnImage.WriteToFile("output_gpu.bmp");

	// Free OpenCL resources
	CL_CALL(clReleaseKernel(bilinear));
	CL_CALL(clReleaseProgram(program));
	CL_CALL(clReleaseCommandQueue(cmdQueue));
	CL_CALL(clReleaseMemObject(dinput));
	CL_CALL(clReleaseMemObject(doutput));
	CL_CALL(clReleaseContext(context));
	free(platforms);
	free(devices);
 
	return 0;
}

