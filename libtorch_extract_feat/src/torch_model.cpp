#include "../include/torch_model.hpp"

int TorchFeature::nomalizeVector(std::vector<float> &v, const int feature_dim)
{
    if (v.size() !=feature_dim) return 0;
    std::vector<float> v_norm;
    float norm_v = sqrt(std::inner_product( v.begin(), v.end(), v.begin(), 0.0 ));
    float denorm_v = std::max(1e-12, (double)norm_v);
    for (auto it = v.begin(); it != v.end(); it++)
    {
        *it = (*it)/denorm_v;
    }
    return 1;
}

int TorchFeature::initModel(const std::string& modelBin)
{
    try {
        module_ = torch::jit::load(modelBin);
    }

    catch (std::exception &e) {
        std::cerr << "Couldn't load weights file, please check the weights_file path!" << std::endl;
        return 0;
    }
    module_->to(device_);
    return 1;
}

int TorchFeature::computeFeat(const cv::Mat &image, std::vector<float> &ft)
{
    if (image.empty()) return 0;
    ft.clear();
    at::Tensor inputTensor = torch::from_blob(image.data,
            {1, image.rows, image.cols, 3}, torch::kFloat32);
    inputTensor = inputTensor.permute({0, 3, 1, 2});
    inputTensor = inputTensor.to(device_);
    torch::Tensor outsTensor = module_->forward({inputTensor}).toTensor();

    for (int k = 0; k < outsTensor.size(1); k++) {
        float value = outsTensor[0][k].item<float>();
        ft.push_back(value);
    }
    int status = nomalizeVector(ft, outsTensor.size(1));
    if (status < 0) {
        return 0;
    }
    return 1;
}

int TorchFeature::computeFeatsBatch(const std::vector<cv::Mat> &imageList, std::vector<std::vector<float>> &fts)
{
    fts.clear();
    const int channels = imageList[0].channels();
    std::vector<at::Tensor> inputsTuple;
    size_t batchNum =  imageList.size()/batchSize_ + 1;
    for (size_t batch = 0; batch < batchNum; batch++) {
        inputsTuple.clear();
        for (size_t i = batch*batchSize_; i < (batch+1)*batchSize_; i++) {
            if (i > imageList.size()-1)  break;
            /*if (imageList[i].empty()) {
                cv::Mat tmpImage = cv::Mat::zeros(netInputSize_, netInputSize_, CV_8UC3);
            }*/
            at::Tensor inputTensor = torch::from_blob(imageList[i].data,
                    {1, imageList[i].rows, imageList[i].cols, 3}, torch::kFloat32);
            inputTensor = inputTensor.permute({0, 3, 1, 2});
            inputTensor = inputTensor.to(device_);
            inputsTuple.emplace_back(inputTensor);
        }

         // Concatenate a batch of tensors
        at::Tensor inputs = torch::cat(inputsTuple, 0);

        torch::Tensor outsTensor = module_->forward({inputs}).toTensor();

        for (int j = 0; j < outsTensor.size(0); j++) {
            std::vector<float> feat;
            for (int k = 0; k < outsTensor.size(1); k++) {
                float value = outsTensor[j][k].item<float>();
                feat.push_back(value);
            }
            nomalizeVector(feat, outsTensor.size(1));
            fts.push_back(feat);
        }
    }
    return 1;
}
