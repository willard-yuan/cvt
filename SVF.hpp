
#ifndef SVF_hpp
#define SVF_hpp

#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#define PI 3.141592653589793f
const static uint32_t kMaxOctaveSlices = 8;

class SVF
{
    
public:
    SVF();
    int getInliers(std::vector<cv::KeyPoint> &qKpts1, std::vector<cv::KeyPoint> &qKpts2,\
                   std::vector<cv::DMatch> &goodMatches);
    std::vector<cv::DMatch> getSVFMatches(); // should be called after getInliers
private:
    std::vector<cv::DMatch> SVFMatches;
    int spaceValidate(const cv::KeyPoint &pa0, const cv::KeyPoint &pa1, \
                      const cv::KeyPoint &pb0, const cv::KeyPoint &pb1);
    
    int OctaveDiff(float octave1, float octave2, int* idx);
    int OctaveValidate(std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, std::vector<cv::DMatch>* good_matches);
    
};

#endif /* SVF_hpp */
