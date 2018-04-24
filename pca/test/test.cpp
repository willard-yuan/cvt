
#include <iostream>
#include "pca_dimension.h"

template<typename Out>
void split(const std::string &s, char delim, Out result, std::string &frame_id) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    int i = 0;
    while (std::getline(ss, item, delim)) {
        if (i == 1){
            frame_id = item;
        }
        if (i > 1){
            *(result++) = std::stof(item);
        }
        ++i;
    }
}

std::vector<float> split(const std::string &s, char delim, std::string &frame_id) {
    std::vector<float> elems;
    split(s, delim, std::back_inserter(elems), frame_id);
    return elems;
}

int main(int argc, const char * argv[]) {
    
    // 读取特征
    std::string line;
    std::string frame_id;
    std::string feats_path = "/Users/willard/codes/cpp/pca_online/pca_online/10.feats";
    std::ifstream fin(feats_path);
    std::vector<std::vector<float>> feats;
    int i = 0;
    while (std::getline(fin, line)){
        std::vector<float> data = split(line, ' ', frame_id);
        feats.push_back(data);
        ++i;
    }
    
    // 转成array
    float* data = new float[feats.size()*1024];
    for(int i = 0; i < feats.size(); i++){
        for(int j = 0; j < 1024; j++){
            data[i*1024+j] = feats[i][j];
        }
    }
    
    // 载入PCA model
    std::string model_path = "/Users/willard/codes/cpp/pca_online/pca_online/pca_128_300w.yml";
    PCAModel model(model_path);
    
    // 降维
    cv::Mat reducedData = model.reductDimension(data, (int)feats.size(), 1024);
    std::cout << reducedData << std::endl;
    
    return 0;
}
