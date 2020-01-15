#ifndef torch_model_hpp
#define torch_model_hpp

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace std;

class TorchFeature{
public:
    TorchFeature(const int batchSize, const std::string modelBin) {
        netInputSize_ = 112;
        batchSize_ = batchSize;
        initModel(modelBin);
    }
    
    ~TorchFeature() {
    }

    int computeFeat(const cv::Mat &image, std::vector<float> &ft);
    int computeFeatsBatch(const std::vector<cv::Mat> &img, std::vector<std::vector<float>> &fts);
    
private:
    int nomalizeVector(std::vector<float> &v, const int feature_dim);
    int initModel(const std::string& modelBin);
    
protected:
    int netInputSize_;
    int batchSize_;

    torch::DeviceType deviceType_ = torch::kCPU;
    torch::Device device_ = torch::Device(deviceType_, -1);
    
    torch::Tensor inputTensor_;
    std::shared_ptr<torch::jit::script::Module> module_;
};

#endif /* torch_model_hpp */
