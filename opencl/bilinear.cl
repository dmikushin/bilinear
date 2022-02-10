typedef struct
{
	unsigned char Red, Green, Blue, Alpha;
}
RGBApixel;

typedef union
{
	RGBApixel p;
	int i;
}
RGBApixel_;

static inline void interpolate(
	__global const RGBApixel* pixels, __global RGBApixel* output, int width, float x, float y)
{
	int px = (int)x; // floor of x
	int py = (int)y; // floor of y
	const int stride = width;
	__global const RGBApixel* p0 = &pixels[0] + px + py * stride; // pointer to first pixel

	// Load the four neighboring pixels
	RGBApixel_ p1_; p1_.i = *(__global int*)&p0[0 + 0 * stride];
	RGBApixel_ p2_; p2_.i = *(__global int*)&p0[1 + 0 * stride];
	RGBApixel_ p3_; p3_.i = *(__global int*)&p0[0 + 1 * stride];
	RGBApixel_ p4_; p4_.i = *(__global int*)&p0[1 + 1 * stride];

	const RGBApixel* p1 = &p1_.p;
	const RGBApixel* p2 = &p2_.p;
	const RGBApixel* p3 = &p3_.p;
	const RGBApixel* p4 = &p4_.p;
	
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
	int outr = p1->Red * w1 + p2->Red * w2 + p3->Red * w3 + p4->Red * w4;
	int outg = p1->Green * w1 + p2->Green * w2 + p3->Green * w3 + p4->Green * w4;
	int outb = p1->Blue * w1 + p2->Blue * w2 + p3->Blue * w3 + p4->Blue * w4;
	int outa = p1->Alpha * w1 + p2->Alpha * w2 + p3->Alpha * w3 + p4->Alpha * w4;

	RGBApixel ret;
	ret.Red = (outr + 128) >> 8;
	ret.Green = (outg + 128) >> 8;
	ret.Blue = (outb + 128) >> 8;
	ret.Alpha = (outa + 128) >> 8;

	__global RGBApixel_* output_ = (__global RGBApixel_*)output;
	output_->p = ret;
}

__kernel void bilinear (const int width, const int height,
	__global RGBApixel* input, __global RGBApixel* output)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	if (j >= 2 * height) return;
	if (i >= 2 * width) return;

	float x = width * (i - 0.5f) / (float)(2 * width);
	float y = height * (j - 0.5f) / (float)(2 * height);

	interpolate(input, &output[i + j * 2 * width], width, x, y);
}

