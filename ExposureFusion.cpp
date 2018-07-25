/**
@brief CUDA real time exposure fusion class
@author ShaneYuan
@date Jul 18, 2018
*/

#include "ExposureFusion.h"

#define MAX_CUDA_SIZE 16

PyramidCUDA::PyramidCUDA() {};
PyramidCUDA::~PyramidCUDA() {};

int PyramidCUDA::buildPyramidLaplacian(cv::cuda::GpuMat img,
	std::vector<cv::cuda::GpuMat> & pyrImgsd,
	std::vector<cv::Mat> & pyrImgsh,
	std::vector<bool> & devices,
	int levels) {
	pyrImgsd[0] = img;
	devices[0] = true;
	for (int i = 0; i < levels - 1; ++i) {
		if (pyrImgsd[i].cols > MAX_CUDA_SIZE &&
			pyrImgsd[i].rows > MAX_CUDA_SIZE) {
			cv::cuda::pyrDown(pyrImgsd[i], pyrImgsd[i + 1]);
			devices[i] = true;
		}
		else if (devices[i - 1] == true) {
			pyrImgsd[i].download(pyrImgsh[i]);
			cv::pyrDown(pyrImgsh[i], pyrImgsh[i + 1]);
			devices[i] = false;
		}
		else {
			cv::pyrDown(pyrImgsh[i], pyrImgsh[i + 1]);
		}
	}
	cv::cuda::GpuMat tmp;
	cv::Mat tmph;
	for (int i = 0; i < levels - 1; ++i) {
		if (devices[i] == true) {
			cv::cuda::pyrUp(pyrImgsd[i + 1], tmp);
			cv::cuda::subtract(pyrImgsd[i], tmp, pyrImgsd[i]);
		}
		else {
			cv::pyrUp(pyrImgsh[i + 1], tmph);
			cv::subtract(pyrImgsh[i], tmph, pyrImgsh[i]);
		}
	}
	return 0;
}

int PyramidCUDA::buildPyramidGaussian(cv::cuda::GpuMat img,
	std::vector<cv::cuda::GpuMat> & pyrImgsd,
	std::vector<cv::Mat> & pyrImgsh,
	std::vector<bool> & devices,
	int levels) {
	pyrImgsd[0] = img;
	devices[0] = true;
	for (int i = 0; i < levels - 1; ++i) {
		if (pyrImgsd[i].cols > MAX_CUDA_SIZE && 
			pyrImgsd[i].rows > MAX_CUDA_SIZE) {
			cv::cuda::pyrDown(pyrImgsd[i], pyrImgsd[i + 1]);
			devices[i] = true;
		}
		else if (devices[i - 1] == true) {
			pyrImgsd[i].download(pyrImgsh[i]);
			cv::pyrDown(pyrImgsh[i], pyrImgsh[i + 1]);
			devices[i] = false;
		}
		else {
			cv::pyrDown(pyrImgsh[i], pyrImgsh[i + 1]);
		}
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
	weightsPyrd.resize(imgNum);
	weightsPyrh.resize(imgNum);
	devices.resize(layerNum);
	for (size_t i = 0; i < imgNum; i++) {
		weightsPyrd[i].resize(layerNum);
		weightsPyrh[i].resize(layerNum);
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

	this->images.resize(imgNum);
	this->pyrImgsd.resize(imgNum);
	this->pyrImgsh.resize(imgNum);
	this->weightsPyrd.resize(imgNum);
	this->weightsPyrh.resize(imgNum);
	for (size_t i = 0; i < imgNum; i++) {
		this->pyrImgsd[i].resize(layerNum);
		this->pyrImgsh[i].resize(layerNum);
		this->weightsPyrd[i].resize(layerNum);
		this->weightsPyrh[i].resize(layerNum);
	}

	// build weight pyramid
	for (size_t i = 0; i < images.size(); i++) {
		std::vector<cv::Mat> weight_pyr;
		cv::cuda::GpuMat weightd;
		weightd.upload(weights[i]);
		cv::cuda::cvtColor(weightd, weightd, cv::COLOR_GRAY2BGR);
		PyramidCUDA::buildPyramidGaussian(weightd, weightsPyrd[i], weightsPyrh[i], devices, layerNum);
		//buildPyramid(weights[i], weight_pyr, layerNum - 1);
		//for (size_t j = 0; j < weight_pyr.size(); j++) {
		//	weightsPyr[i][j].upload(weight_pyr[j]);
		//	cv::cuda::cvtColor(weightsPyr[i][j], weightsPyr[i][j], cv::COLOR_GRAY2BGR);
		//}
	}

	resPyrd.resize(layerNum);
	resPyrh.resize(layerNum);
	for (size_t j = 0; j < layerNum; j++) {
		cv::Size size;
		if (devices[j] == true)
			size = weightsPyrd[0][j].size();
		else size = weightsPyrh[0][j].size();
		resPyrd[j].create(size, CV_32FC3);
		resPyrh[j].create(size, CV_32FC3);
	}

	darkf.create(dark.size(), CV_32FC3);
	lightf.create(light.size(), CV_32FC3);

	return 0;
}

/**
@brief fusion two images
@return int
*/
int ExposureFusion::fusion(cv::cuda::GpuMat dark, cv::cuda::GpuMat light,
	cv::cuda::GpuMat & fusion) {
	dark.convertTo(darkf, CV_32FC3);
	light.convertTo(lightf, CV_32FC3);
	PyramidCUDA::buildPyramidLaplacian(darkf, pyrImgsd[0], pyrImgsh[0], devices, layerNum);
	PyramidCUDA::buildPyramidLaplacian(lightf, pyrImgsd[1], pyrImgsh[1], devices, layerNum);

	for (size_t j = 0; j < layerNum; j++) {
		if (j == 0) {
			resPyrd[j].setTo(cv::Scalar(0, 0, 0));
		}
		else if (devices[j] == true && devices[j - 1] == true) {
			resPyrd[j].setTo(cv::Scalar(0, 0, 0));
		}
		else if (devices[j] == false && devices[j - 1] == true) {
			resPyrd[j].setTo(cv::Scalar(0, 0, 0));
			resPyrh[j].setTo(cv::Scalar(0, 0, 0));
		}
		else {
			resPyrh[j].setTo(cv::Scalar(0, 0, 0));
		}
	}

	for (size_t i = 0; i < 2; i++) {
		for (int lvl = 0; lvl < layerNum; lvl++) {
			if (devices[lvl] == true) {
				cv::cuda::multiply(pyrImgsd[i][lvl], weightsPyrd[i][lvl], pyrImgsd[i][lvl]);
				cv::cuda::add(resPyrd[lvl], pyrImgsd[i][lvl], resPyrd[lvl]);
			}
			else {
				cv::multiply(pyrImgsh[i][lvl], weightsPyrh[i][lvl], pyrImgsh[i][lvl]);
				cv::add(resPyrh[lvl], pyrImgsh[i][lvl], resPyrh[lvl]);
			}
		}
	}
	for (int lvl = layerNum - 1; lvl > 0; lvl--) {
		if (devices[lvl] == false && devices[lvl - 1] == false) {
			cv::Mat up;
			cv::pyrUp(resPyrh[lvl], up);
			cv::add(resPyrh[lvl - 1], up, resPyrh[lvl - 1]);
		}
		else if (devices[lvl] == false && devices[lvl - 1] == true){
			cv::Mat up;
			cv::pyrUp(resPyrh[lvl], up);
			cv::add(resPyrh[lvl - 1], up, resPyrh[lvl - 1]);
			resPyrd[lvl - 1].upload(resPyrh[lvl - 1]);
		}
		else {
			cv::cuda::GpuMat up;
			cv::cuda::pyrUp(resPyrd[lvl], up);
			cv::cuda::add(resPyrd[lvl - 1], up, resPyrd[lvl - 1]);
		}
	}
	fusion = resPyrd[0];
	return 0;
}