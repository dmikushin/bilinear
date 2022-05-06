#include <chrono>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "EasyBMP.h"

#define CUDA_CALL(x) do { cudaError_t err = x; if (( err ) != cudaSuccess ) { \
	printf ("Error \"%s\" at %s :%d \n" , cudaGetErrorString(err), \
		__FILE__ , __LINE__ ) ; exit(-1); \
}} while (0)

using namespace std;
using namespace std::chrono;

#define ARR(T, i, j) (T[(i) + (j) * width])

__device__ inline void interpolate(
	const RGBApixel* pixels, RGBApixel& output, int width, float x, float y)
{
	int px = (int)x; // floor of x
	int py = (int)y; // floor of y
	const int stride = width;
	const RGBApixel* p0 = &pixels[0] + px + py * stride; // pointer to first pixel

	// load the four neighboring pixels
	const RGBApixel& p1 = p0[0 + 0 * stride];
	const RGBApixel& p2 = p0[1 + 0 * stride];
	const RGBApixel& p3 = p0[0 + 1 * stride];
	const RGBApixel& p4 = p0[1 + 1 * stride];

	// Calculate the weights for each pixel
	float fx = x - px;
	float fy = y - py;
	float fx1 = 1.0f - fx;
	float fy1 = 1.0f - fy;

	int w1 = fx1 * fy1 * 256.0f + 0.5f;
	int w2 = fx  * fy1 * 256.0f + 0.5f;
	int w3 = fx1 * fy  * 256.0f + 0.5f;
	int w4 = fx  * fy  * 256.0f + 0.5f;

	// Calculate the weighted sum of pixels (for each color channel)
	int outr = p1.Red * w1 + p2.Red * w2 + p3.Red * w3 + p4.Red * w4;
	int outg = p1.Green * w1 + p2.Green * w2 + p3.Green * w3 + p4.Green * w4;
	int outb = p1.Blue * w1 + p2.Blue * w2 + p3.Blue * w3 + p4.Blue * w4;
	int outa = p1.Alpha * w1 + p2.Alpha * w2 + p3.Alpha * w3 + p4.Alpha * w4;

	output.Red = (outr + 128) >> 8;
	output.Green = (outg + 128) >> 8;
	output.Blue = (outb + 128) >> 8;
	output.Alpha = (outa + 128) >> 8;
}

__global__ void bilinear (const int width, const int height,
	RGBApixel* input, RGBApixel* output)
{
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (j >= 2 * height) return;
	if (i >= 2 * width) return;

	float x = width * (i - 0.5) / (float)(2 * width);
	float y = height * (j - 0.5) / (float)(2 * height);

	interpolate(input, output[i + j * 2 * width], width, x, y);
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

	RGBApixel *dinput, *doutput;
	CUDA_CALL(cudaMalloc(&dinput, sizeof(RGBApixel) * input.size()));
	CUDA_CALL(cudaMalloc(&doutput, sizeof(RGBApixel) * output.size()));
	CUDA_CALL(cudaMemcpy(dinput, &input[0], sizeof(RGBApixel) * input.size(), cudaMemcpyHostToDevice));

	auto start = high_resolution_clock::now();

	dim3 szblock(128, 1, 1);
	dim3 nblocks(2 * width / szblock.x, 2 * height, 1);
	if (2 * width % szblock.x) nblocks.x++;
	bilinear<<<nblocks, szblock>>>(width, height, dinput, doutput);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

	auto finish = high_resolution_clock::now();

	cout << "GPU kernel time = " <<
		duration_cast<milliseconds>(finish - start).count() <<
		" ms" << endl;

	CUDA_CALL(cudaMemcpy(&output[0], doutput, sizeof(RGBApixel) * output.size(), cudaMemcpyDeviceToHost));

	AnImage.SetSize(2 * width, 2 * height);
	for (int i = 0; i < 2 * width; i++)
		for (int j = 0; j < 2 * height; j++)
			AnImage.SetPixel(i, j, output[i + j * 2 * width]);

	AnImage.WriteToFile("output_gpu.bmp");
 
	return 0;
}

