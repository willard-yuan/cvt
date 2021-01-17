#include "../include/dupNet.h"

#include <string>
#include <algorithm>
#include <vector>

//#include "glog/logging.h"

int CnnFeature::nomalizeVector(std::vector<float> &v, const int feature_dim) {
    if (v.size() !=feature_dim) return 0;
    std::vector<float> v_norm;
    float norm_v = sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    float denorm_v = std::max(1e-12, (double)norm_v);
    for (auto it = v.begin(); it != v.end(); it++) {
        *it = (*it)/denorm_v;
    }
    return 1;
}

int CnnFeature::initModel(const std::string& modelTxt, const std::string& modelBin) {
    std::ifstream modelTxtFile(modelTxt);
    std::ifstream modelBinFile(modelBin);
    if (!modelTxtFile.good() || !modelBinFile.good()) {
      //LOG(ERROR) << "model file does not existed" << std::endl;
      return 0;
    }

    dnnNet = new cv::dnn::Net(cv::dnn::readNetFromCaffe(modelTxt, modelBin));
    dnnNet->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    dnnNet->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    std::vector<std::string> layers = dnnNet->getLayerNames();
    /*for (int i = 0; i < layers.size(); ++i) {
      std::cout << "layer: " << layers[i] << std::endl;
    }*/

    std::vector<cv::Mat> imgs;
    cv::Mat img(netInputSize, netInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < batchSize; i++) {
        imgs.push_back(img.clone());
    }
    cv::Mat inputBlob = cv::dnn::blobFromImages(imgs, 1.0f, cv::Size(netInputSize, netInputSize),
                                                cv::Scalar(104, 117, 123), false, false);
    dnnNet->setInput(inputBlob, "data");
    cv::Mat featMat = dnnNet->forward(blobName);

    return 1;
}

int CnnFeature::computeFeat(const cv::Mat& img, std::vector<float> &ft) {
    if (img.empty()) {
      //LOG(INFO) << "image is empty:" << std::endl;
      return 0;
    }

    ft.clear();
    cv::Mat imgResized;
    cv::resize(img, imgResized, cv::Size(netInputSize, netInputSize));
    cv::Mat inputBlob = cv::dnn::blobFromImage(imgResized, 1.0f, cv::Size(),
                                               cv::Scalar(104, 117, 123), false, false);
    dnnNet->setInput(inputBlob, "data");
    cv::Mat featMat = dnnNet->forward(blobName);

    for (int i = 0; i < featMat.size[1]; i++) {
    // for (int i = 0; i < featMat.cols; i++) {
        ft.push_back(featMat.at<float>(0, i));
    }
    nomalizeVector(ft, (int)ft.size());
    return 1;
}

int CnnFeature::computeFeatsBatch(const std::vector<cv::Mat> &imgs, std::vector<std::vector<float>> &fts) {
    if (imgs.empty()) {
        //LOG(INFO) << "images are empty:" << std::endl;
        return 0;
    }

    fts.clear();

    int nBatchs = (int)imgs.size() / batchSize;
    int residNum = (int)imgs.size() % batchSize;

    std::vector<cv::Mat> imgsBatch;
    cv::Mat inputBlob;
    std::vector<float> ft;

    // n 个 batch
    for (int i = 0; i < nBatchs; i++) {
        imgsBatch.clear();
        cv::Mat imgResized;
        for (int j = 0; j < batchSize; j++) {
          if (imgs.at(i*batchSize+j).empty()) {
            //LOG(INFO) << "images are empty:" << std::endl;
            imgResized = cv::Mat::zeros(netInputSize, netInputSize, CV_8UC3);
          } else {
            cv::resize(imgs.at(i*batchSize+j), imgResized, cv::Size(netInputSize, netInputSize));
          }
          imgsBatch.push_back(imgResized.clone());  // 深拷贝，非常重要
        }

        // dupNet
        inputBlob = cv::dnn::blobFromImages(imgsBatch, 1.0f, cv::Size(),
                cv::Scalar(104, 117, 123), false, false);
        dnnNet->setInput(inputBlob, "data");
        cv::Mat featMat = dnnNet->forward(blobName);
        // auto featMat = dnnNet->forward();
        /*std::cout << featMat.size[0] << "x" << featMat.size[1] << "x" << featMat.size[2]
            << "x" << featMat.size[3] << std::endl;*/

        /*std::cout << featMat.cols << "*" << featMat.rows << "*" << featMat.channels() << std::endl; */

        for (int m = 0; m < featMat.size[0]; m++) {
            ft.clear();
            for (int k = 0; k < featMat.size[1]; k++) {
                ft.push_back(featMat.at<float>(m, k));
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
          //LOG(INFO) << "images are empty:" << std::endl;
          tmpImg = cv::Mat::zeros(netInputSize, netInputSize, CV_8UC3);
        }
        if (computeFeat(tmpImg, ft)) {
            fts.push_back(ft);
        }
    }

    return 1;
}
