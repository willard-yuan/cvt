#ifndef erase_specular_h
#define erase_specular_h

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <stdio.h>

void erase_specular(cv::Mat& eye_grey);

void test_erase_specular();

#endif /* __hpp */
