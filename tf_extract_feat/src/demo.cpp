#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tfExtractFeature.hpp"

int main(int argc, char* argv[]) {
    
    int batchSize = 1;
    string modelBinPath = "../model/pb.model";
    string imagePath = "../data/1019.png";
    
    CnnFeature* model = new CnnFeature(batchSize, modelBinPath);
    
    cv::Mat img = cv::imread(imagePath, 1);
    
    std::vector<float> feat;
    int status = model->computeFeat(img, feat);
    if (status != 1)
    {
        return 0;
    }
        
    for (int i = 0; i < feat.size(); i++)
    {
        std::cout << feat.at(i) << " ";
    }
    std::cout << std::endl;

    return 0;
}
