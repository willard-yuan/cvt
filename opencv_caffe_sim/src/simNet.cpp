
#include "simNet.hpp"

#include <string>

int CnnFeature::nomalizeVector(std::vector<float> &v, const int feature_dim)
{
    if (v.size() !=feature_dim) return 0;
    std::vector<float> v_norm;
    float norm_v = sqrt(std::inner_product( v.begin(), v.end(), v.begin(), 0.0 ));
    float denorm_v = std::max(1e-12, (double)norm_v);
    for (auto it = v.begin(); it != v.end(); it++){
        *it = (*it)/denorm_v;
    }
    return 1;
}

int CnnFeature::initModel(const std::string& modelTxt, const std::string& modelBin)
{
    std::ifstream modelTxtFile(modelTxt);
    std::ifstream modelBinFile(modelBin);
    if (!modelTxtFile.good() || !modelBinFile.good()) {
        std::cout << "model file does not existed" << std::endl;
        return 0;
    }

    dnnNet = new cv::dnn::Net(cv::dnn::readNetFromCaffe(modelTxt, modelBin));
    dnnNet->setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    
    std::vector<cv::Mat> imgs;
    cv::Mat img(netInputSize, netInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
    for(int i = 0; i < batchSize; i++)
    {
        imgs.push_back(img.clone());
    }
    
    cv::Mat inputBlob = cv::dnn::blobFromImages(imgs, 1.0f, cv::Size(netInputSize, netInputSize),
                                                cv::Scalar(103.72267235, 116.14597303, 122.05719166), false, false);
    /*cv::Mat inputBlob = cv::dnn::blobFromImages(imgs, 1.0f, cv::Size(netInputSize, netInputSize),
                                               cv::Scalar(104.08953193177008, 115.80300577828721, 121.36268150129982), false, false);*/
    
    dnnNet->setInput(inputBlob, "data");
    cv::Mat featMat = dnnNet->forward(blobName);
    
    return 1;
}

int CnnFeature::computeFeat(const cv::Mat& img, std::vector<float> &ft)
{
    
    if (img.empty()) {
        return 0;
    }
    
    ft.clear();
    cv::Mat imgResized;
    cv::resize(img, imgResized, cv::Size(netInputSize, netInputSize));
    
    
    // simNet: mean_value: B 103.72267235, G 116.14597303, R 122.05719166
    // dupNet: mean_value: 104.08953193177008, 115.80300577828721, 121.36268150129982
    
    cv::Mat inputBlob = cv::dnn::blobFromImage(imgResized, 1.0f, cv::Size(netInputSize, netInputSize),
                                                cv::Scalar(103.72267235, 116.14597303, 122.05719166), false, false);
    /*cv::Mat inputBlob = cv::dnn::blobFromImage(imgResized, 1.0f, cv::Size(netInputSize, netInputSize),
                                        //cv::Scalar(104.08953193177008, 115.80300577828721, 121.36268150129982), false, false);*/
    
    
    
    dnnNet->setInput(inputBlob, "data");
    cv::Mat featMat = dnnNet->forward(blobName);
    
    for(int i = 0; i < featMat.cols; i++)
    {
        ft.push_back(featMat.at<float>(0, i));
    }
    nomalizeVector(ft, (int)ft.size());
    return 1;
}

int CnnFeature::computeFeatsBatch(const std::vector<cv::Mat> &imgs, std::vector<std::vector<float>> &fts)
{
    if (imgs.empty()) {
        return 0;
    }
    
    fts.clear();

    int nBatchs = (int)imgs.size() / batchSize;
    int residNum = (int)imgs.size() % batchSize;
    
    std::vector<cv::Mat> imgsBatch;
    std::vector<float> ft;
    
    // n 个batch
    for (int i = 0; i < nBatchs; i++) {
        imgsBatch.clear();
        cv::Mat imgResized;
        for (int j = 0; j < batchSize; j++) {
            if (imgs.at(i*batchSize+j).empty()) {
                std::cout << "image is empty" << std::endl;
                imgResized = cv::Mat::zeros(netInputSize, netInputSize, CV_8UC3);
            } else {
                cv::resize(imgs.at(i*batchSize+j), imgResized, cv::Size(netInputSize, netInputSize));
            }
            imgsBatch.push_back(imgResized.clone());  // 深拷贝，非常重要
        }
        
        // simNet: mean_value: 103.72267235, 116.14597303, 122.05719166
        // dupNet: mean_value: 104.08953193177008, 115.80300577828721, 121.36268150129982
        
        cv::Mat inputBlob = cv::dnn::blobFromImages(imgsBatch, 1.0f, cv::Size(netInputSize, netInputSize),
                                                    cv::Scalar(103.72267235, 116.14597303, 122.05719166), false, false);
        //cv::Mat inputBlob = cv::dnn::blobFromImages(imgsBatch, 1.0f, cv::Size(netInputSize, netInputSize),
                                            //cv::Scalar(104.08953193177008, 115.80300577828721, 121.36268150129982), false, false);

        dnnNet->setInput(inputBlob, "data");
        cv::Mat featMat = dnnNet->forward("fc1");
        
        for(int n = 0; n < featMat.rows; n++)
        {
            ft.clear();
            for(int k = 0; k < featMat.cols; k++)
            {
                ft.push_back(featMat.at<float>(n, k));
            }
            nomalizeVector(ft, (int)ft.size());
            fts.push_back(ft);
        }
    }
    
    // 剩余的图片
    for (int i = 0; i < residNum; i++) {
        ft.clear();
        cv::Mat tmpImg = imgs.at(nBatchs*batchSize + i);
        if (tmpImg.empty()) {
            std::cout << "image is empty" << std::endl;
            tmpImg = cv::Mat::zeros(netInputSize, netInputSize, CV_8UC3);
        }
        if (computeFeat(tmpImg, ft)) {
            fts.push_back(ft);
        }
    }
    
    return 1;
}

