
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
std::vector<float> nomalize_vecotor(std::vector<float> &v) {
    std::vector<float> v_norm;
    float norm_v = sqrt(std::inner_product( v.begin(), v.end(), v.begin(), 0.0 ));
    float denorm_v = std::max(1e-12, (double)norm_v);
    for (auto it = v.begin(); it != v.end(); it++){
        float tmp = (*it)/denorm_v;
        v_norm.push_back(tmp);
    }
    return v_norm;
}

int main() {

    std::string training_data_path = "/raid/yuanyong/ann/faiss_sq/cpp/1b_64d_feats.bin";
    std::string model_path = "../model/int8_sq_model.bin";

    int d = 64;  // dimension

    std::ifstream fp;
    int num_db = 0;
    fp.open(training_data_path.c_str(), std::ios::in|std::ios::binary);
    fp.read((char*)&num_db, sizeof(int));

    std::cout << "db num: " << num_db << std::endl;

    //float *xb = new float[d * num_db];
    float *xb = new float[d * 30000000];

    int count = 0;
    float *feat = new float[d];
    std::vector<std::string> dbIds;
    for (int i = 0; i < 30000000; i++) {
    //for (int i = 0; i < num_db; i++) {
        // 读取id
        int idSize = 0;
        fp.read((char*)&idSize, sizeof(int));
        char idName[1024] = {""};
        fp.read(idName, idSize);
        std::string idStr(idName);

        //std::cout << "idName: " << idStr << std::endl;

        // 读取特征
        int dim_feat = 0;
        fp.read((char*)&dim_feat, sizeof(int));
        if (dim_feat != d) {
            cout << "file error";
            exit(1);
        }
        
        fp.read(reinterpret_cast<char *>(feat), dim_feat*sizeof(float));
        std::vector<float> tmp_feat(feat, feat + d);
        std::vector<float> tmp_feat_normed = nomalize_vecotor(tmp_feat);

        for (int j = 0; j < dim_feat; j++) {
            xb[d * i + j] = tmp_feat_normed[j];
            //xb[d * i + j] = tmp_feat[j];
        }

        dbIds.push_back(idStr);
        if (count % 1000000 == 0) {
            std::cout << "read num: " << count << std::endl;
        }
        ++count;
    }

    faiss::IndexScalarQuantizer SQuantizer(d, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2);
    //SQuantizer.train(num_db, xb);
    SQuantizer.train(30000000, xb);
    //SQuantizer.add(nb, xb);
    faiss::write_index(&SQuantizer, model_path.c_str());
    

    float *xtest = new float[d];
    uint8_t *bytes = new uint8_t[d];
    float *xtest_decode = new float[d];

    memcpy(xtest, xb, d*sizeof(float));
    SQuantizer.sq.compute_codes(xtest, bytes, 1);

    float *xtest_decode = new float[d];
    SQuantizer.sq.decode(bytes, xtest_decode, 1);

    std::cout << "原始值(64维) " << "int8压缩后表示 " << "解码后表示" << std::endl;
    float loss = 0.0
    for (int i = 0; i < 64; i++) {
        std::cout << xb[i] << ", " << unsigned(bytes[i]) << ", " << xtest_decode[i] << std::endl;
        loss + = (xb[i] - xtest_decode[i])*(xb[i] - xtest_decode[i]);
    }

    float average_loss = loss/1.0

    delete [] xb;
    delete [] bytes;
    delete [] xtest;

    return 0;
}
