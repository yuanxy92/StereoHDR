/**
@brief main.cpp
@author ShaneYuan
@date Jul 18, 2018
*/

#include "ExposureFusion.h"

int main(int argc, char* argv[]) {
	std::vector<cv::Mat> images;
	images.push_back(cv::imread("E:\\Project\\StereoHDR\\data\\1.jpg"));
	images.push_back(cv::imread("E:\\Project\\StereoHDR\\data\\3.jpg"));

	cv::Mat fusion;
	cv::Ptr<cv::MergeMertens> merge_mertens = cv::createMergeMertens();
	merge_mertens->process(images, fusion);

	cv::imwrite("E:\\Project\\StereoHDR\\data\\fusion.jpg", fusion * 255);

	return 0;
}