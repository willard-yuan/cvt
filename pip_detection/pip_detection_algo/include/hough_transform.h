#pragma once

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace mmu {

struct LineDetectInfo{
  std::vector<float> xSelectcoords;
  std::vector<float> ySelectcoords;
};

class HoughTransform{

public:
    enum Type {
        STANDARD,
        PROBABILISTIC
    };
    
    HoughTransform();
    LineDetectInfo detectLines(cv::Mat & input_frame, const std::string & frameId);

private:
    HoughTransform::Type type;
    int hough_transform_threshold_max;
    int hough_transform_threshold;
    float imgScale;
};

} //end namespace mmu
