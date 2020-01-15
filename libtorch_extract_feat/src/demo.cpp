#include <glob.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include "../include/torch_model.hpp"

std::vector<std::string> globVector(const std::string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    std::vector<std::string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        auto tmp = std::string(glob_result.gl_pathv[i]);
        files.push_back(tmp);
    }
    std::sort(files.begin(), files.end());
    globfree(&glob_result);
    return files;
}

bool LoadImage(std::string file_name, cv::Mat &image) {
  image = cv::imread(file_name, 1);
  if (image.empty() || !image.data) {
    return false;
  }
  cv::resize(image, image, cv::Size(112, 112));
  image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
  cv::Mat meanMat(112, 112, CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));
  image = (image - meanMat)/0.5;
  return true;
}

#if 0
int main(int argc, const char *argv[]) {
    std::string imagePath = "./data/late.png";
    std::string modelPath = "./net.pt";

    std::vector<float> feature;
    cv::Mat tmpImage;
    LoadImage(imagePath, tmpImage);
    TorchFeature model = TorchFeature(1, modelPath);
    model.computeFeat(tmpImage, feature);

    std::cout << imagePath << std::endl;
    for (int j = 0; j < feature.size(); j++) {
        std::cout << feature.at(j) << " ";
    }
    std::cout << std::endl;
}
#endif

#if 1
int main(int argc, const char *argv[]) {
    int batchSize = 8;
    std::string imagesDir = "./data";
    std::string modelPath = "./net.pt";

    std::vector<cv::Mat> imageList;
    std::vector<std::vector<float>> features;

    std::vector<std::string> imagesPath = globVector(imagesDir + + "/*.*");

    for(int i = 0; i < imagesPath.size(); i++) {
        cv::Mat tmpImage;
        bool status = LoadImage(imagesPath.at(i), tmpImage);
        imageList.push_back(tmpImage);
    }

    TorchFeature model = TorchFeature(batchSize, modelPath);
    model.computeFeatsBatch(imageList, features);

    for(int i = 0; i < features.size(); i++)
    {
        std::cout << imagesPath.at(i) << std::endl;
        for (int j = 0; j < features.at(i).size(); j++)
        {
            std::cout << features.at(i).at(j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

}
#endif
