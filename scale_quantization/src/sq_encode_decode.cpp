
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include <faiss/Index.h>
#include <faiss/index_io.h>
#include <faiss/IndexScalarQuantizer.h>

using namespace std;
using namespace std::chrono;

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

    std::string training_data_path = "/raid/yuanyong/ann/faiss_sq/cpp/tmp_10000_64d_feats.bin";
    std::string model_path = "../model/int8_sq_model.bin";

    int num_test = 10000;

    faiss::Index * index_model = faiss::read_index(model_path.c_str());
    faiss::IndexScalarQuantizer * SQuantizer = reinterpret_cast<faiss::IndexScalarQuantizer*>(index_model);

    int d = 64;  // dimension

    std::ifstream fp;
    int num_db = 0;
    fp.open(training_data_path.c_str(), std::ios::in|std::ios::binary);
    fp.read((char*)&num_db, sizeof(int));

    std::cout << "db num: " << num_db << std::endl;

    float *xb = new float[d * num_test];

    int count = 0;
    std::vector<std::string> dbIds;

    float *feat = new float[d];
    float *xtest = new float[d];
    uint8_t *bytes = new uint8_t[d];
    float *xtest_decode = new float[d];

    float loss = 0;

    double encode_time = 0.0;
    double decode_time = 0.0;
    for (int i = 0; i < num_test; i++) {
        int idSize = 0;
        fp.read((char*)&idSize, sizeof(int));
        char idName[1024] = {""};
        fp.read(idName, idSize);
        std::string idStr(idName);

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
            xtest[j] = tmp_feat_normed[j];
        }
        
        //auto start = high_resolution_clock::now();
        SQuantizer->sq.compute_codes(xtest, bytes, 1);
        //auto stop = high_resolution_clock::now();
        //auto duration = duration_cast<microseconds>(stop - start);
        //cout << "\nTime taken by function : "<< duration.count() << " microseconds";
        auto start = high_resolution_clock::now();
        SQuantizer->sq.decode(bytes, xtest_decode, 1);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "\nTime taken by function : "<< duration.count() << " microseconds";
        
        //std::cout << "原始值(64维) " << "int8压缩后表示 " << "解码后表示" << std::endl;
        for (int i = 0; i < 64; i++) {
            //std::cout << tmp_feat_normed[i] << ", " << unsigned(bytes[i]) << ", " << xtest_decode[i] << std::endl;
            loss += (tmp_feat_normed[i]-xtest_decode[i])*(tmp_feat_normed[i]-xtest_decode[i]);
        }

        /*for (int i = 0; i < 64; i++) {
            std::cout << tmp_feat_normed[i] << ",";
        }
        std::cout << std::endl;
        for (int i = 0; i < 64; i++) {
            std::cout << unsigned(bytes[i]) << ",";
        }
        std::cout << std::endl;
        for (int i = 0; i < 64; i++) {
            std::cout << xtest_decode[i] << ",";
        }
        std::cout << std::endl;*/
    }

    std::cout << "average loss: " << loss/num_test << std::endl;
    std::cout << "average encode time: " << encode_time/num_test << std::endl;
    std::cout << "average decode time: " << decode_time/num_test << std::endl;

    delete [] feat;
    delete [] xtest_decode;
    delete [] bytes;
    delete [] xtest;

    return 0;
}
