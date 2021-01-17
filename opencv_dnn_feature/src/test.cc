#include "../include/dupNet.h"
#include "../include/baseUtils.h"

int main(int argc, char **argv) {
    cv::String modelTxt = "../model/deploy_imgl_batch_test.prototxt";
    cv::String modelBin = "../model/imgl.model";
    int batchSize = 8;

    cv::String imagesFile = "../data/*.*";

    CnnFeature cnnFeatComputor = CnnFeature(batchSize, modelTxt, modelBin);
    std::vector<std::string> imgsPath = globVector(imagesFile);

    std::vector<cv::Mat> imgs;
    std::vector<std::string> baseNames;

    for (int i = 0; i < imgsPath.size(); i++) {
        cv::Mat img = cv::imread(imgsPath.at(i), 1);
        if (img.empty()) continue;

        imgs.push_back(img);

        std::string baseName = removeExtension(getBaseName(imgsPath.at(i), "/"));
        baseNames.push_back(baseName);
        printf("%d(%d) %s\n", i+1, (int)imgsPath.size(), baseName.c_str());
    }

    // 测试
    std::vector<std::vector<float>> ftsTmp;
    cnnFeatComputor.computeFeatsBatch(imgs, ftsTmp);
    for (int i = 0; i < ftsTmp.size(); i++) {
        std::cout << baseNames.at(i) << " ";
        for (int j= 0; j < ftsTmp.at(0).size(); j++) {
            std::cout << ftsTmp.at(i).at(j) << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
