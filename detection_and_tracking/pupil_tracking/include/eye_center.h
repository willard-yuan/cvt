#ifndef eye_center_h
#define eye_center_h

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat get_centermap(cv::Mat& eye_grey);

cv::Point find_eye_center(cv::Mat& eye_bgr);

void test_centermap();


#endif /* eye_center_hpp */
