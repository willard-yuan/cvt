#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"

namespace cvtk {
class CannyEdgeDetector {
public:
 CannyEdgeDetector();
    cv::Mat detectEdges(cv::Mat & input_frame, cv::Mat & canny_edge_detecion_frame);
private:
    int low_threshold_max;
    int high_threshold_max;
    int low_slider_position;
    int high_slider_position;
};
} // end namespce cvtk
