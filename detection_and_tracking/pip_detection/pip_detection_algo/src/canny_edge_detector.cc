#include "../include/canny_edge_detector.h"

namespace cvtk {

CannyEdgeDetector::CannyEdgeDetector() {
  low_threshold_max = 500;
  high_threshold_max = 500;
  low_slider_position = 0;
  high_slider_position = 200;
}

 cv::Mat CannyEdgeDetector::detectEdges(cv::Mat & input_frame, cv::Mat & canny_edge_detecion_frame) {
  cv::cvtColor(input_frame, canny_edge_detecion_frame, CV_RGB2GRAY);
  cv::Canny(canny_edge_detecion_frame, canny_edge_detecion_frame, low_slider_position, high_slider_position,3);
  return canny_edge_detecion_frame;
}

} // end namespace cvtk
