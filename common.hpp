
#ifndef common_hpp
#define common_hpp

#include <stdio.h>

#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

void drawMatch(cv::Mat &srcImg, cv::Mat &objImg, const std::vector<cv::KeyPoint> &srcPts, const std::vector<cv::KeyPoint> &dstPts);

int ResizeImage(const cv::Mat& raw_img, cv::Mat& img, const int kShorterEdge, bool resize);

#endif /* common_hpp */
