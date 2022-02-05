#pragma once

#include <string>
#include <stdio.h>
#include <vector>
#include <numeric>

#include "glog/logging.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "protos/cvtk_video_edge_detection.pb.h"

#include "canny_edge_detector.h"
#include "hough_transform.h"

namespace cvtk {
namespace detect {
struct Frame {
  std::string id;  // frame id: photoId_frameId
  std::vector<char> image;  // 图片
};

struct Video {
  std::string id;
  std::vector<Frame> frames;
};
        
struct MotionAreaInfo {
  int left;
  int right;
  int top;
  int bottom;
};
        
class MotionAreaDetecion {
 public:
  MotionAreaDetecion();
  ~MotionAreaDetecion(){};
  int MotionAreaDetect(const Video& Video, ::cvtk::img::MotionArea* motionLocInfo);
  int PipDetect(const Video& Video, ::cvtk::img::MotionArea* motionLocInfo);
 protected:
  cv::Ptr<cv::BackgroundSubtractor> pBackSub;
  cv::Mat kernel;
  float imgScale;
  int thresh;
  int maxValue;
  float motionAreaRatio;
  float motionAreaRatioLow;
 private:
  bool invalidWidthAndHeight(const cv::Mat& mat);
  int globalThreshold(cv::Mat &image, cv::Mat &imageThresd, int setThresh=70);
  int findMaxBBox(cv::Mat& outMask, cv::Rect& maxBBox);
  int findTopKBBox(cv::Mat& outMask, std::vector<cv::Rect>& BBoxs);
  CannyEdgeDetector cannyEdgeDetector;
  HoughTransform houghTransform;
};
};
}
