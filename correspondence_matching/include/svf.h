#pragma once

#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

typedef unsigned char uint8;

std::vector<cv::DMatch> getInliers(std::vector<cv::KeyPoint> &qKpts1, 
	std::vector<cv::KeyPoint> &qKpts2, std::vector<cv::DMatch> &rawMatches,
	bool removeRepeat);
int spaceValidate(const cv::KeyPoint &pa0, 
	const cv::KeyPoint &pa1, const cv::KeyPoint &pb0, const cv::KeyPoint &pb1);
uint8 getSiteCode(int height, int width, cv::Point2f pt);
int getGlobalFeature(cv::Mat img, std::vector<float> &fea);
float distanceL2(const std::vector<float> &x, const std::vector<float> &y);
float distanceL2(const cv::KeyPoint &x, const cv::KeyPoint &y);
void rootSift(cv::Mat &descriptors, const float eps = 1e-7);
