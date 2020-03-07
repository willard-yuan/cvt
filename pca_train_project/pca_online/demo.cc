#include <iostream>
#include "pca_utils.h"

template<typename Out>
void split(const std::string &s, char delim, Out result, std::string &frame_id) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    int i = 0;
    while (std::getline(ss, item, delim)) {
        if (i == 0){
            frame_id = item;
        }
        if (i >= 1){
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
    
    int dim_feat = 2048;
    
    // 读取特征
    std::string line;
    std::string frame_id;
    std::string feats_path = "data.txt";
    std::ifstream fin(feats_path);
    std::vector<std::string> names;
    std::vector<std::vector<float>> feats;
    int i = 0;
    while (std::getline(fin, line)){
        std::vector<float> data = split(line, ',', frame_id);
        feats.push_back(data);
        names.push_back(frame_id);
        ++i;
    }
    
    // 转成array
    float* data = new float[feats.size()*dim_feat];
    for(int i = 0; i < feats.size(); i++){
        for(int j = 0; j < dim_feat; j++){
            data[i*dim_feat+j] = feats[i][j];
        }
    }
    
    // 载入PCA model
    std::string model_path = "./model/pca_256_500w.yml";
    mmu::PCAUtils model = *mmu::PCAUtils::getInstance();
    model.loadModel(model_path);
    
    // 降维
    cv::Mat reduceMat;
    model.reduceDim(data, (int)feats.size(), dim_feat, reduceMat);
    for(int i = 0; i < feats.size(); i++) {
        std::cout << names.at(i) << ": " << reduceMat.row(i) << std::endl;
    }
    
    delete [] data;
    
    return 0;
}
