
#ifndef GHT_hpp
#define GHT_hpp

#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "common.hpp"

#define PI 3.141592653589793f
const static uint32_t kMaxOctaveSlices = 8;

class GHT
{
public:
    GHT();
    int getInliers(std::vector<cv::KeyPoint> &qKpts1, std::vector<cv::KeyPoint> &qKpts2,\
                   std::vector<cv::DMatch> &goodMatches);
    std::vector<cv::DMatch> getSVFMatches(); // should be called after getInliers
    
    
    void getRansacInliers(std::vector<cv::KeyPoint> &qKeypoints, std::vector<cv::KeyPoint> &objKeypoints, std::vector<cv::DMatch> &matches, const cv::Mat &srcColorImage, const cv::Mat &dstColorImage);
private:
    std::vector<cv::DMatch> SVFMatches;
    int spaceValidate(const cv::KeyPoint &pa0, const cv::KeyPoint &pa1, \
                      const cv::KeyPoint &pb0, const cv::KeyPoint &pb1);
    
    int OctaveDiff(float octave1, float octave2, int* idx);
    int OctaveValidate(std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, std::vector<cv::DMatch>* good_matches);
    
    std::vector<cv::DMatch> delRepeat(std::vector<cv::DMatch> & rawMatches);
    
    void plotMatches(const cv::Mat &srcColorImage, const cv::Mat &dstColorImage,
                     std::vector<cv::Point2f> &srcPoints, std::vector<cv::Point2f> &dstPoints);
    
    cv::Scalar randomColor(cv::RNG& rng);
};

#endif /* SVF_hpp */
