#pragma once

#include <stdio.h>

#include <numeric>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

class CnnFeature{
 public:
    CnnFeature(const int batchSize_, const std::string modelTxt_, const std::string modelBin_) {
        dnnNet = NULL;
        netInputSize = 224;
        // blobName = "fc1";
        blobName = "pool5/7x7_s1";
        // blobName = "pool5/drop_7x7_s1";
        batchSize = batchSize_;
        initModel(modelTxt_, modelBin_);
    }

    ~CnnFeature() {
        delete dnnNet;
    }
    int computeFeat(const cv::Mat& img, std::vector<float> &ft);  // NOLINT
    int computeFeatsBatch(const std::vector<cv::Mat> &img, std::vector<std::vector<float>> &fts);  // NOLINT

 private:
    int nomalizeVector(std::vector<float> &v, const int feature_dim);  // NOLINT
    int initModel(const std::string& modelTxt, const std::string& modelBin);

// protected:
    int netInputSize;
    int batchSize;
    cv::dnn::Net* dnnNet;
    std::string blobName;
};
