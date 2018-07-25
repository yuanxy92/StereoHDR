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

int main(int argc, char* argv[]) {
	std::vector<cv::Mat> images;
	images.push_back(cv::imread("E:\\Project\\StereoHDR\\data\\1.jpg"));
	images.push_back(cv::imread("E:\\Project\\StereoHDR\\data\\2.jpg"));

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

	ExposureFusion exFusion;
	exFusion.calcWeight(images[0], images[1]);
	cv::cuda::GpuMat dark, light, fusion;
	dark.upload(images[0]);
	light.upload(images[1]);
	exFusion.fusion(dark, light, fusion);

	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	exFusion.fusion(dark, light, fusion);
	//for (int i = 0; i < 1000; i ++) {
	//	exFusion.fusion(dark, light, fusion);
	//}

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU exposure fusion step: (file:%s, line:%d) elapsed time : %f ms\n", __FILE__, __LINE__, elapsedTime);


	cv::Mat fusion_d;
	fusion.download(fusion_d);

	return 0;
}