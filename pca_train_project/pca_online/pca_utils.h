#ifndef PCA_UTILS_H_
#define PCA_UTILS_H_

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <string>

namespace cvtk {

class PCAUtils {
 public:
  static PCAUtils* instance_;

  ~PCAUtils() {
  }

  static PCAUtils* getInstance() {
    static PCAUtils inst;
    return &inst;
  }

  void loadModel(const std::string& filename);

  void reduceDim(const float* data, int num, int dim, cv::Mat& reduceMat);  // NOLINT
  void reduceDim(const cv::Mat& mat, cv::Mat& reduceMat);  // NOLINT
  cv::Mat reduceDim(const cv::Mat& mat);

 private:
  void convertToMat(const float* data, int num, int dim, cv::Mat& mat);  // NOLINT

  cv::PCA pca_;
};

}  // namespace cvtk

#endif  // PCA_UTILS_H_
