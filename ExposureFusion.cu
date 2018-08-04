#include "ExposureFusion.h"

// cuda
// CUDA includes
#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>

// CUDA utilities and system includes
//#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/opencv.hpp>

//from:https://github.com/aashikgowda/Bilateral-Filter-CUDA/blob/master/kernel.cu
#include <iostream>
#include <algorithm>
#include <ctime>
#include <opencv2/opencv.hpp>

#define M_PI           3.14159265358979323846
#define TILE_X 16
#define TILE_Y 16


using namespace std;
using namespace cv;
// 1D Gaussian kernel array values of a fixed size (make sure the number > filter size d)
__constant__ float cGaussian[64];
// Initialize texture memory to store the input
// texture<unsigned char, 2, cudaReadModeElementType> inTexture;
/*
GAUSSIAN IN 1D FOR SPATIAL DIFFERENCE
Here, exp(-[(x_centre - x_curr)^2 + (y_centre - y_curr)^2]/(2*sigma*sigma)) can be broken down into ...
exp[-(x_centre - x_curr)^2 / (2*sigma*sigma)] * exp[-(y_centre - y_curr)^2 / (2*sigma*sigma)]
i.e, 2D gaussian -> product of two 1D Gaussian
A constant Gaussian 1D array can be initialzed to store the gaussian values
Eg: For a kernel size 5, the pixel difference array will be ...
[-2, -1, 0, 1 , 2] for which the gaussian kernel is applied
*/
void updateGaussian(int r, float sd)
{
	float fGaussian[64];
	for (int i = 0; i < 2 * r + 1; i++)
	{
		float x = i - r;
		fGaussian[i] = expf(-(x*x) / (2 * sd*sd));
	}
	cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2 * r + 1));
}

// Gaussian function for range difference
__device__ inline float gaussian(float x, float sigma)
{
	return __expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

__global__ void cuda_bilateralFilter(cv::cuda::PtrStep<float> input, cv::cuda::PtrStep<float> output, int width, int height,
	int r, float sI, float sS)
{
	// Initialize global Tile indices along x,y and xy
	int txIndex = __mul24(blockIdx.x, TILE_X) + threadIdx.x;
	int tyIndex = __mul24(blockIdx.y, TILE_Y) + threadIdx.y;

	// If within image size
	if ((txIndex < width) && (tyIndex < height))
	{
		float iFiltered = 0;
		float wP = 0;
		float centrePx = input(tyIndex, txIndex);
		for (int dy = -r; dy <= r; dy++) 
		{
			for (int dx = -r; dx <= r; dx++) 
			{
				float currPx = (txIndex + dx < width && txIndex + dx >= 0 && tyIndex + dy < height && tyIndex + dy >= 0) ?
					(input(tyIndex + dy, txIndex + dx)) : (input(tyIndex, txIndex));//(0.0f);
				// Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Range or Intensity difference)
				float w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian(centrePx - currPx, sI);
				iFiltered += w * currPx;
				wP += w;
			}
		}
		output(tyIndex, txIndex) = iFiltered / wP;
	}
}

__global__ void cuda_fusionRaman(cv::cuda::PtrStep<uchar3> input1, cv::cuda::PtrStep<uchar3> input2,
	cv::cuda::PtrStep<uchar3> input1_XYZ, cv::cuda::PtrStep<uchar3> input2_XYZ,
	cv::cuda::PtrStep<uchar3> output, int width, int height,
	float C,int r, float sI, float sS)
{
	// Initialize global Tile indices along x,y and xy
	int txIndex = __mul24(blockIdx.x, TILE_X) + threadIdx.x;
	int tyIndex = __mul24(blockIdx.y, TILE_Y) + threadIdx.y;

	// If within image size
	if ((txIndex < width) && (tyIndex < height))
	{
		uchar3 ip1 = input1(tyIndex, txIndex);
		uchar3 ip2 = input2(tyIndex, txIndex);
		//unsigned char lum_ip1 = input1_XYZ(tyIndex, txIndex).y;
		//unsigned char lum_ip2 = input2_XYZ(tyIndex, txIndex).y;
		//float lum = 0.2126f * ip1.x + 0.7152 * ip1.y + 0.0722 * ip1.z; //0~255
		float weight1 = 0.0f;
		float weight2 = 0.0f;
		float total = 0.0f;

		{
			float iFiltered = 0;
			float wP = 0;
			float centrePx = input1_XYZ(tyIndex, txIndex).y;
			for (int dy = -r; dy <= r; dy++)
			{
				for (int dx = -r; dx <= r; dx++)
				{
					float currPx = (txIndex + dx < width && txIndex + dx >= 0 && tyIndex + dy < height && tyIndex + dy >= 0) ?
						(input1_XYZ(tyIndex + dy, txIndex + dx).y) : (0.0f);

					float w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian(centrePx - currPx, sI);
					iFiltered += w * currPx;
					wP += w;
				}
			}
			weight1 = fabs(iFiltered / wP - centrePx) / 255.0f + C;
		}
		{
			float iFiltered = 0;
			float wP = 0;
			float centrePx = input2_XYZ(tyIndex, txIndex).y;
			for (int dy = -r; dy <= r; dy++)
			{
				for (int dx = -r; dx <= r; dx++)
				{
					float currPx = (txIndex + dx < width && txIndex + dx >= 0 && tyIndex + dy < height && tyIndex + dy >= 0) ?
						(input2_XYZ(tyIndex + dy, txIndex + dx).y) : (0.0f);

					float w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian(centrePx - currPx, sI);
					iFiltered += w * currPx;
					wP += w;
				}
			}
			weight2 = fabs(iFiltered / wP - centrePx) / 255.0f + C;
		}
		total = weight1 + weight2;

		unsigned char ansx = (unsigned char)((weight1 * ip1.x + weight2 * ip2.x) / total);
		unsigned char ansy = (unsigned char)((weight1 * ip1.y + weight2 * ip2.y) / total);
		unsigned char ansz = (unsigned char)((weight1 * ip1.z + weight2 * ip2.z) / total);

		//printf("FromCudaKernel::ansxyz = %d %d %d\n", ansx,ansy,ansz);

		output(tyIndex, txIndex) = make_uchar3(ansx, ansy, ansz);

	}
}

int ExposureFusion::fusionRaman_kernal(cv::cuda::GpuMat &img1, cv::cuda::GpuMat &img2, cv::cuda::GpuMat & ret)
{
	int radius = 1;

	int height = img1.rows;
	int width = img1.cols;
	int col = img1.channels();
	int n = 2;
	ret.create(height, width, CV_8UC3);

	float C = 70.0 / 255.0;
	float K1 = 1.0;
	float K2 = 1.0 / 10.0;
	float sigma_s = MIN(height, width);
	float imageStackMax = 255.0;//TODO:should search the img. ref:https://docs.opencv.org/3.4/d5/de6/group__cudaarithm__reduce.html#ga5cacbc2a2323c4eaa81e7390c5d9f530
	float imageStackMin = 0.0;
	float sigma_r = K2 * (imageStackMax - imageStackMin);
	updateGaussian(radius, sigma_s); // TODO: Only once, fix me

	cv::cuda::GpuMat XYZ1, XYZ2;
	//XYZ1.create(height, width, CV_32FC3);
	//XYZ2.create(height, width, CV_32FC3);
	cv::cuda::cvtColor(img1, XYZ1, cv::COLOR_RGB2XYZ);
	cv::cuda::cvtColor(img2, XYZ2, cv::COLOR_RGB2XYZ);

	dim3 block(TILE_X, TILE_Y);
	//Calculate grid size to cover the whole image
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	cuda_fusionRaman << <grid, block >> > (img1, img2, XYZ1, XYZ2, ret, width, height, C, radius, sigma_r, sigma_s);

	//cv::Mat t;
	//ret.download(t);

	return 0;
}




int ExposureFusion::bilateralFilter_GPU(cv::cuda::GpuMat & input, cv::cuda::GpuMat & output, int r, float sI, float sS)
{
	updateGaussian(r, sS); // TODO: fix me
	dim3 block(TILE_X, TILE_Y);
	//Calculate grid size to cover the whole image
	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	cuda_bilateralFilter << <grid, block >> > (input, output, input.cols, input.rows, r, sI, sS);
	return 0;
}