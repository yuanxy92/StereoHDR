/**
@brief CUDA real time exposure fusion class
@author ShaneYuan
@date Jul 18, 2018
*/

#include "ExposureFusion.h"

#include <cuda.h>
#include <cuda_runtime.h>

PyramidCUDA::PyramidCUDA() {};
PyramidCUDA::~PyramidCUDA() {};

int PyramidCUDA::buildPyramidLaplacian(cv::cuda::GpuMat img,
	std::vector<cv::cuda::GpuMat> & pyrImgs, int levels) {
	img.convertTo(pyrImgs[0], CV_32F);
	for (int i = 0; i < levels; ++i) {
		cv::cuda::pyrDown(pyrImgs[i], pyrImgs[i + 1]);
	}
	cv::cuda::GpuMat tmp, tmp2;
	for (int i = 0; i < levels; ++i) {
		cv::cuda::pyrUp(pyrImgs[i + 1], tmp);
		cv::cuda::subtract(pyrImgs[i], tmp, pyrImgs[i]);
	}
	return 0;
}

ExposureFusion::ExposureFusion() {
	wcon = 1.0f;
	wsat = 1.0f;
	wexp = 0.0f;
}
ExposureFusion::~ExposureFusion() {}

/**
@brief calculate weight mat to blend hdr result
@return int
*/
int ExposureFusion::calcWeight(cv::Mat dark, cv::Mat light) {
	// prepare variables
	imgNum = 2;
	layerNum = 11;
	weights.resize(imgNum);
	weightsPyr.resize(imgNum);
	for (size_t i = 0; i < imgNum; i++) {
		weightsPyr[i].resize(layerNum);
	}
	std::vector<cv::Mat> images(imgNum);
	images[0] = dark;
	images[1] = light;
	cv::Size size = images[0].size();
	cv::Mat weight_sum(size, CV_32F);
	weight_sum.setTo(0);

	// calcualte weights
	int channels = 3;
	for (size_t i = 0; i < images.size(); i++) {
		cv::Mat img, gray, contrast, saturation, wellexp;
		std::vector<cv::Mat> splitted(channels);

		images[i].convertTo(img, CV_32F, 1.0f / 255.0f);
		if (channels == 3) {
			cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
		}
		else {
			img.copyTo(gray);
		}
		cv::split(img, splitted);

		cv::Laplacian(gray, contrast, CV_32F);
		contrast = cv::abs(contrast);

		cv::Mat mean = cv::Mat::zeros(size, CV_32F);
		for (int c = 0; c < channels; c++) {
			mean += splitted[c];
		}
		mean /= channels;

		saturation = cv::Mat::zeros(size, CV_32F);
		for (int c = 0; c < channels; c++) {
			cv::Mat deviation = splitted[c] - mean;
			cv::pow(deviation, 2.0f, deviation);
			saturation += deviation;
		}
		cv::sqrt(saturation, saturation);

		wellexp = cv::Mat::ones(size, CV_32F);
		for (int c = 0; c < channels; c++) {
			cv::Mat expo = splitted[c] - 0.5f;
			cv::pow(expo, 2.0f, expo);
			expo = -expo / 0.08f;
			cv::exp(expo, expo);
			wellexp = wellexp.mul(expo);
		}

		cv::pow(contrast, wcon, contrast);
		cv::pow(saturation, wsat, saturation);
		cv::pow(wellexp, wexp, wellexp);

		weights[i] = contrast;
		if (channels == 3) {
			weights[i] = weights[i].mul(saturation);
		}
		weights[i] = weights[i].mul(wellexp) + 1e-12f;
		weight_sum += weights[i];
	}
	for (size_t i = 0; i < images.size(); i++) {
		weights[i] /= weight_sum;
	}

	// build weight pyramid
	for (size_t i = 0; i < images.size(); i++) {
		std::vector<cv::Mat> weight_pyr;
		buildPyramid(weights[i], weight_pyr, layerNum - 1);
		for (size_t j = 0; j < weight_pyr.size(); j++) {
			weightsPyr[i][j].upload(weight_pyr[j]);
			cv::cuda::cvtColor(weightsPyr[i][j], weightsPyr[i][j], cv::COLOR_GRAY2BGR);
		}
	}

	this->images.resize(imgNum);
	this->pyrImgs.resize(imgNum);
	for (size_t i = 0; i < imgNum; i++) {
		this->pyrImgs[i].resize(layerNum);
	}

	return 0;
}

/**
@brief fusion two images
@return int
*/
int ExposureFusion::fusion(cv::cuda::GpuMat dark, cv::cuda::GpuMat light,
	cv::cuda::GpuMat & fusion) {
	images[0] = dark;
	images[1] = light;
	PyramidCUDA::buildPyramidLaplacian(images[0], pyrImgs[0], layerNum - 1);
	PyramidCUDA::buildPyramidLaplacian(images[1], pyrImgs[1], layerNum - 1);

	if (resPyr.size() == 0) {
		resPyr.resize(pyrImgs[0].size());
		for (size_t j = 0; j < layerNum; j++) {
			resPyr[j].create(pyrImgs[0][j].size(), CV_32FC3);
		}
	}
	for (size_t j = 0; j < layerNum; j++) {
		resPyr[j].setTo(cv::Scalar(0, 0, 0));
	}

	for (size_t i = 0; i < images.size(); i++) {
		for (int lvl = 0; lvl < layerNum; lvl++) {
			cv::cuda::multiply(pyrImgs[i][lvl], weightsPyr[i][lvl], pyrImgs[i][lvl]);
			cv::cuda::add(resPyr[lvl], pyrImgs[i][lvl], resPyr[lvl]);
		}
	}
	for (int lvl = layerNum - 1; lvl > 0; lvl--) {
		cv::cuda::GpuMat up;
		cv::cuda::pyrUp(resPyr[lvl], up);
		cv::cuda::add(resPyr[lvl - 1], up, resPyr[lvl - 1]);
	}
	fusion = resPyr[0];
	return 0;
}

int ExposureFusion::fusionRaman(cv::cuda::GpuMat dark, cv::cuda::GpuMat light, cv::cuda::GpuMat & fusion, int code)
{
	std::vector<cv::cuda::GpuMat> imgStack;
	imgStack.push_back(dark);
	imgStack.push_back(light);
	this->fusionRaman(imgStack, fusion, code);
	return 0;
}

int ExposureFusion::fusionRaman(std::vector<cv::cuda::GpuMat> imgStack, cv::cuda::GpuMat & fusion, int code)
{

	if (imgStack.size() < 2)
		return -1;

	std::vector<cv::cuda::GpuMat> useageStack;
	useageStack.resize(imgStack.size());
	for (int i = 0; i < imgStack.size(); i++)
	{
		imgStack[i].convertTo(useageStack[i], CV_32FC3, 1.0/255);
	}
	
	int r = imgStack[0].rows;
	int c = imgStack[0].cols;
	int col = imgStack[0].channels();
	int n = imgStack.size();

	double C = 70.0 / 255.0;
	double K1 = 1.0;
	double K2 = 1.0 / 10.0;
	double sigma_s = MIN(r, c);
	double imageStackMax = 1.0;//TODO:should search the img ref:https://docs.opencv.org/3.4/d5/de6/group__cudaarithm__reduce.html#ga5cacbc2a2323c4eaa81e7390c5d9f530
	double imageStackMin = 0.0;
	double sigma_r = K2 * (imageStackMax - imageStackMin);
	cv::cuda::GpuMat total;
	std::vector<cv::cuda::GpuMat> weight;
	weight.resize(n);
	total.create(r, c, CV_32FC1);
	total.setTo(cv::Scalar(0));



	for (int i = 0; i < n; i++)
	{
		
		weight[i].create(r, c, CV_32FC1);
		cv::cuda::GpuMat XYZ;
		std::vector<cv::cuda::GpuMat> split_XYZ;
		cv::cuda::cvtColor(useageStack[i], XYZ, code);
		cv::cuda::split(XYZ, split_XYZ);

		cudaEvent_t start, stop;
		float elapsedTime;
		cudaEventCreate(&start);
		cudaEventRecord(start, 0);
		//split_XYZ[1].download(t);

		//cv::cuda::bilateralFilter(split_XYZ[1], weight[i], 3, sigma_r, sigma_s); //TODO: stream maybe?
		this->bilateralFilter_GPU(split_XYZ[1], weight[i], 1, sigma_r, sigma_s);
		//this->bilateralFilter_GPU(split_XYZ[1], weight[i], 1, sigma_s, sigma_r);

		cv::Mat t;
		weight[i].download(t);

		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("fusion: (file:%s, line:%d) elapsed time : %f ms\n", __FILE__, __LINE__, elapsedTime);
		//weight[i].download(t);
		//int x = 10;
		cv::cuda::absdiff(weight[i], split_XYZ[1], weight[i]);
		weight[i].convertTo(weight[i], CV_32FC1, 1.0, C);
		
		cv::cuda::add(weight[i], total, total);
	}





	fusion.create(r, c, CV_32FC3);
	fusion.setTo(cv::Scalar(0,0,0));


	for (int i = 0; i < n; i++)
	{
		cv::cuda::GpuMat tmp,tmpdiv;
		cv::cuda::divide(weight[i], total, tmpdiv);
		cv::cuda::cvtColor(tmpdiv, tmpdiv, cv::COLOR_GRAY2BGR);
		cv::cuda::multiply(useageStack[i], tmpdiv, tmp);
		cv::cuda::add(fusion, tmp, fusion);
	}




	//cv::Mat t;
	//fusion.download(t);
	
	return 0;
}

int ExposureFusion::fusionRaman(std::vector<cv::Mat> imgStack, cv::Mat & fusion, int code)
{
	if (imgStack.size() < 2)
		return -1;

	std::vector<cv::Mat> useageStack;
	useageStack.resize(imgStack.size());
	for (int i = 0; i < imgStack.size(); i++)
	{
		imgStack[i].convertTo(useageStack[i], CV_32FC3, 1.0 / 255);
	}

	int r = imgStack[0].rows;
	int c = imgStack[0].cols;
	int col = imgStack[0].channels();
	int n = imgStack.size();

	double C = 70.0 / 255.0;
	double K1 = 1.0;
	double K2 = 1.0 / 10.0;
	double sigma_s = MIN(r, c);
	double imageStackMax = 1.0;//TODO:should search the img ref:https://docs.opencv.org/3.4/d5/de6/group__cudaarithm__reduce.html#ga5cacbc2a2323c4eaa81e7390c5d9f530
	double imageStackMin = 0.0;
	double sigma_r = K2 * (imageStackMax - imageStackMin);
	cv::Mat total;
	std::vector<cv::Mat> weight;
	weight.resize(n);
	total.create(r, c, CV_32FC1);
	total.setTo(cv::Scalar(0));
	for (int i = 0; i < n; i++)
	{

		weight[i].create(r, c, CV_32FC1);
		cv::Mat XYZ;
		std::vector<cv::Mat> split_XYZ;
		cv::cvtColor(useageStack[i], XYZ, code);
		cv::split(XYZ, split_XYZ);

		//split_XYZ[1].download(t);

		cv::bilateralFilter(split_XYZ[1], weight[i], 3, 0.1, r); //TODO: stream maybe?
																	   //weight[i].download(t);
																	   //int x = 10;
		cv::absdiff(weight[i], split_XYZ[1], weight[i]);
		weight[i].convertTo(weight[i], CV_32FC1, 1.0, C);

		cv::add(weight[i], total, total);
	}
	fusion.create(r, c, CV_32FC3);
	fusion.setTo(cv::Scalar(0, 0, 0));
	for (int i = 0; i < n; i++)
	{
		cv::Mat tmp, tmpdiv;
		cv::divide(weight[i], total, tmpdiv);
		cv::cvtColor(tmpdiv, tmpdiv, cv::COLOR_GRAY2BGR);
		cv::multiply(useageStack[i], tmpdiv, tmp);
		cv::add(fusion, tmp, fusion);
	}
	return 0;
}


