
#ifndef tfExtractFeature_hpp
#define tfExtractFeature_hpp

#include <stdio.h>

#include <numeric>
#include <fstream>
#include <iostream>
#include <cstdlib>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

class CnnFeature{
public:
    CnnFeature(const int batchSize_, const std::string modelBin_) {
        netInputSize = 640;
        inPutName = "input:0";
        outPutName = "head/out_emb:0";
        batchSize = batchSize_;
        initModel(modelBin_);
    }
    
    ~CnnFeature() {
    }
    int computeFeat(const cv::Mat& img, std::vector<float> &ft);
    int computeFeatsBatch(const std::vector<cv::Mat> &img, std::vector<std::vector<float>> &fts);
    
private:
    int nomalizeVector(std::vector<float> &v, const int feature_dim);
    int initModel(const std::string& modelBin);
    
protected:
    int netInputSize;
    int batchSize;
    tensorflow::GraphDef graph_def;
    unique_ptr<tensorflow::Session> session;
    tensorflow::SessionOptions sess_opt;
    
    tensorflow::Tensor inputTensor;
    
    std::string inPutName = "input:0";
    std::string outPutName = "head/out_emb:0";
};

#endif /* tfExtractFeature_hpp */
