
#ifndef simNet_hpp
#define simNet_hpp

#include <stdio.h>

#include <numeric>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class CnnFeature{
public:
    CnnFeature(const int batchSize_, const std::string modelTxt_, const std::string modelBin_) {
        dnnNet = NULL;
        netInputSize = 299;
        blobName = "fc1";
        batchSize = batchSize_;
        initModel(modelTxt_, modelBin_);
    }
    
    ~CnnFeature() {
        delete dnnNet;
    }
    int computeFeat(const cv::Mat& img, std::vector<float> &ft);
    int computeFeatsBatch(const std::vector<cv::Mat> &img, std::vector<std::vector<float>> &fts);
    
private:
    int nomalizeVector(std::vector<float> &v, const int feature_dim);
    int initModel(const std::string& modelTxt, const std::string& modelBin);
    
protected:
    int netInputSize;
    int batchSize;
    cv::dnn::Net* dnnNet;
    std::string blobName;
};

#endif /* simNet_hpp */
