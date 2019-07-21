#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../include/tfExtractFeature.hpp"

int main(int argc, char* argv[]) {

    int batchSize = 4;
    string modelBinPath = "pb.model";
    string imagePath1 = "1.png";
    string imagePath2 = "2.png";
    string imagePath3 = "3.png";
    string imagePath4 = "4.png";

    CnnFeature* model = new CnnFeature(batchSize, modelBinPath);

    // 单张图片测试
    cv::Mat img1 = cv::imread(imagePath1, 1);
    cv::Mat img2 = cv::imread(imagePath2, 1);
    cv::Mat img3 = cv::imread(imagePath3, 1);
    cv::Mat img4 = cv::imread(imagePath4, 1);

    std::vector<float> feat;
    int status = model->computeFeat(img2, feat);

    if (status != 1) return 0;
    for (int i = 0; i < feat.size(); i++)
    {
        std::cout << feat.at(i) << " ";
    }
    std::cout << std::endl;


    // batch 测试
    std::vector<cv::Mat> imgs;
    std::vector<std::vector<float>> feats;
    imgs.push_back(img1);
    imgs.push_back(img2);
    imgs.push_back(img3);
    imgs.push_back(img4);
    status = model->computeFeatsBatch(imgs, feats);
    if (status != 1) return 0;

    for (int i = 0; i < feats.size(); i++)
    {
        for (int j = 0; j < feats.at(i).size(); j++)
        {
            std::cout << feats.at(i).at(j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
