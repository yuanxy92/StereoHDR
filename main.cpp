/**
@brief main.cpp
@author ShaneYuan
@date Jul 18, 2018
*/

#include "ExposureFusion.h"

// cuda
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>

void fusion_roman()
{
	std::vector<cv::Mat> images;
	std::vector<cv::cuda::GpuMat> gpu_images;
	images.push_back(cv::imread("E:\\data\\EF\\1.jpg"));
	images.push_back(cv::imread("E:\\data\\EF\\2.jpg"));
	gpu_images.resize(images.size());
	for (int i = 0; i < images.size(); i++)
		gpu_images[i].upload(images[i]);
	cv::cuda::GpuMat fusion, fusion255;
	ExposureFusion exFusion;

	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	for(int i = 0; i< 10;i++)
		exFusion.fusionRaman(gpu_images, fusion, cv::COLOR_BGR2XYZ);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Exposure Fusion: (file:%s, line:%d) elapsed time : %f ms\n", __FILE__, __LINE__, elapsedTime);

	fusion.convertTo(fusion255, CV_8UC3, 255.0);
	cv::Mat ans;
	fusion255.download(ans);
	cv::imwrite("E://data//EF//fusion.jpg", ans);
	getchar();
	return;
}

void fusion_py()
{
	std::vector<cv::Mat> images;
	images.push_back(cv::imread("E:\\data\\EF\\1.jpg"));
	images.push_back(cv::imread("E:\\data\\EF\\2.jpg"));
	cv::cuda::GpuMat dark, light, fusion;
	ExposureFusion exFusion;
	exFusion.calcWeight(images[0], images[1]);
	//cv::cuda::GpuMat dark, light, fusion;
	dark.upload(images[0]);
	light.upload(images[1]);
	exFusion.fusion(dark, light, fusion);

	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	for (int i = 0; i < 1000; i++) {
		exFusion.fusion(dark, light, fusion);
	}

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU exposure fusion step: (file:%s, line:%d) elapsed time : %f ms\n", __FILE__, __LINE__, elapsedTime);


	cv::Mat fusion_d;
	fusion.download(fusion_d);
}


int main(int argc, char* argv[]) {
	fusion_roman();

	return 0;
}

//cv::Mat fusion;
//cv::Ptr<cv::MergeMertens> merge_mertens = cv::createMergeMertens();
//merge_mertens->process(images, fusion);

//cv::imwrite("E:\\Project\\StereoHDR\\data\\fusion.jpg", fusion * 255);

//std::vector<cv::cuda::GpuMat> pyrImgs;
//std::vector<cv::cuda::GpuMat> imgd(2);
//imgd[0].upload(images[0]);
//imgd[1].upload(images[1]);

//cudaEvent_t start, stop;
//float elapsedTime;
//cudaEventCreate(&start);
//cudaEventRecord(start, 0);

//PyramidCUDA::buildPyramidLaplacian(imgd[0], pyrImgs, 4);
//PyramidCUDA::buildPyramidLaplacian(imgd[1], pyrImgs, 4);

//cudaEventCreate(&stop);
//cudaEventRecord(stop, 0);
//cudaEventSynchronize(stop);
//cudaEventElapsedTime(&elapsedTime, start, stop);
//printf("Build Laplacian pyramid step: (file:%s, line:%d) elapsed time : %f ms\n", __FILE__, __LINE__, elapsedTime);