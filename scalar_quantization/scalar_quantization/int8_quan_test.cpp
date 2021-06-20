#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include "int8_quan.h"

int main() {

    //std::string model_conf_path = "model/local_conf.json";
    /*std::ifstream fin(model_conf_path);
    nlohmann::json model_conf;
    fin >> model_conf;
    fin.close();
    std::vector<faiss::IndexScalarQuantizer*> SQuantizers;
    for (int i = 0; i < model_conf.size(); ++i) {
      std::cout << model_conf[std::to_string(i)] << std::endl;
      std::string tmp_model_path = model_conf[std::to_string(i)]["model_path"];
      auto SQuantizer = reinterpret_cast<faiss::IndexScalarQuantizer*>(faiss::read_index(tmp_model_path.c_str()));
      SQuantizers.emplace_back(SQuantizer);
    }*/

    std::string model_path = "model/int8_siamese_photo_embedding_8kw.bin";
    float xtmp[64] = {0.7678224, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.6331244, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.583638, 0.76271933, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21529453, 0.0, 0.0, 1.2015152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88310665, 0.0, 0.0, 0.19277531, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5779805, 0.0, 0.0, 0.7728174, 0.0, 2.21898, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    mmu::search::Int8Quan Int8QuanModel = mmu::search::Int8Quan(model_path);
    //mmu::search::Int8Quan Int8QuanModel = mmu::search::Int8Quan(model_conf_path, 2);
    float xtest[64];
    uint8_t bytes[64];

    for (int i = 0; i < 64; i++) {
        xtest[i] = xtmp[i];
    }

    int status = Int8QuanModel.Int8Encode(xtest, bytes, 64, false, 0);

    std::string tmp_string(bytes, bytes + 64);
    std::cout << "encoding string: " << tmp_string << std::endl;

    float xtest_decode[64];
    status = Int8QuanModel.Int8Decode(tmp_string, xtest_decode);

    float inner_product = 0.0;
    std::cout << "原始值(64维) " << std::endl;
    for (int i = 0; i < 64; i++) {
        std::cout << xtest[i] << " ";
        inner_product += xtest[i]*xtest_decode[i];
    }
    std::cout << std::endl;

    std::cout << "int8压缩后表示: " << std::endl;
    for (int i = 0; i < 64; i++) {
      std::cout << unsigned(bytes[i]) << " ";
    }
    std::cout << std::endl;

    std::cout << "解码后表示: " << std::endl;
    for (int i = 0; i < 64; i++) {
      std::cout << xtest_decode[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "inner_product: " << inner_product << std::endl;
    return 0;
}
