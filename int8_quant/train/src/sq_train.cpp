#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>

#include <faiss/Index.h>
#include <faiss/index_io.h>
#include <faiss/IndexScalarQuantizer.h>

using namespace std;

// Normalize vector by L2 norm
std::vector<float> nomalize_vector(std::vector<float> &v) {
    std::vector<float> v_norm;
    float norm_v = sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    float denorm_v = std::max(1e-12, (double)norm_v);
    for (auto it = v.begin(); it != v.end(); it++){
        float tmp = (*it)/denorm_v;
        v_norm.emplace_back(tmp);
    }
    return v_norm;
}

void L2NomalizeVector(float* vector, int d) {
  double accum = 0.0;
  for (int i = 0; i < d; ++i) {
    accum += vector[i] * vector[i];
  }
  accum = sqrt(accum);
  float denorm_v = std::max((double)1e-12, (double)accum);
  for (int i = 0; i < d; ++i) {
    vector[i] = vector[i]/denorm_v;
  }
}

int main() {

    std::string training_data_path = "/raid/yuanyong/ann/faiss_sq/cpp/1b_64d_feats.bin";
    std::string model_path = "../model/int8_sq_model_v2_1b.bin";

    size_t d = 64;  // dimension
    size_t num_db = 0;

    std::ifstream fp;
    fp.open(training_data_path.c_str(), std::ios::in|std::ios::binary);
    fp.read((char*)&num_db, sizeof(int));

    std::cout << "db num: " << num_db << std::endl;

    float *xb = new float[d*num_db];
    float *feat = new float[d];

    int count = 0;
    std::vector<std::string> dbIds;
    for (size_t i = 0; i < num_db; ++i) {  // size_t 很重要，数据量太大避免溢出
        int idSize = 0;
        char idName[1024] = {""};
        std::string idStr;
        fp.read((char*)&idSize, sizeof(int));
        fp.read(idName, idSize);
        idStr = std::string(idName);

        int dim_feat = 0;
        fp.read((char*)&dim_feat, sizeof(int));
        if (dim_feat != d) {
          std::cout << "file error: " << dim_feat << std::endl;
          exit(1);
        }
        
        fp.read(reinterpret_cast<char *>(feat), dim_feat*sizeof(float));
        L2NomalizeVector(feat, dim_feat);

        for (size_t j = 0; j < dim_feat; ++j) {
          xb[dim_feat*i + j] = feat[j];
        }

        dbIds.push_back(idStr);
        if (count % 1000000 == 0) {
            std::cout << "read num: " << count << std::endl;
        }
        ++count;
    }
    fp.close();

    faiss::IndexScalarQuantizer SQuantizer(d, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2);
    SQuantizer.train(num_db, xb);
    // SQuantizer.add(num_db, xb);    
    faiss::write_index(&SQuantizer, model_path.c_str());
    

    float *xtest = new float[d];
    uint8_t *bytes = new uint8_t[d];
    float *xtest_decode = new float[d];

    memcpy(xtest, xb, d*sizeof(float));
    SQuantizer.sq.compute_codes(xtest, bytes, 1);

    SQuantizer.sq.decode(bytes, xtest_decode, 1);

    std::cout << "原始值(64维) " << "int8压缩后表示 " << "解码后表示" << std::endl;
    float loss = 0.0;
    float inner_product = 0.0;
    for (int i = 0; i < 64; i++) {
        std::cout << xb[i] << ", " << unsigned(bytes[i]) << ", " << xtest_decode[i] << std::endl;
        loss += (xb[i] - xtest_decode[i])*(xb[i] - xtest_decode[i]);
        inner_product += xtest[i]*xtest_decode[i];
    }

    float average_loss = loss/1.0;
    std::cout << "average_loss: " << average_loss << std::endl;
    std::cout << "inner_product: " << inner_product << std::endl;

    delete [] xb;
    delete [] feat;
    delete [] bytes;
    delete [] xtest;
    delete [] xtest_decode;

    return 0;
}
