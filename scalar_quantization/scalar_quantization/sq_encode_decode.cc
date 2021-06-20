#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include "third_party/faiss/include/Index.h"
#include "third_party/faiss/include/index_io.h"
#include "third_party/faiss/include/IndexScalarQuantizer.h"

// Normalize vector by L2 norm
std::vector<float> nomalize_vector(std::vector<float> &v) {  //  NOLINT
    std::vector<float> v_norm;
    float norm_v = sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    float denorm_v = std::max(1e-12, (double)norm_v);
    for (auto it = v.begin(); it != v.end(); it++) {
        float tmp = (*it)/denorm_v;
        v_norm.emplace_back(tmp);
    }
    return v_norm;
}

void L2NomalizeVector(float* vector, int d) {  // NOLINT
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
    std::string training_data_path = "1000_test.bin";
    std::string model_path = "int8_sq_model.bin";

    faiss::Index * index_model = faiss::read_index(model_path.c_str());
    faiss::IndexScalarQuantizer * SQuantizer = reinterpret_cast<faiss::IndexScalarQuantizer*>(index_model);

    int d = 64;  // dimension

    auto trained_result = SQuantizer->sq.trained;

    std::ifstream fp;
    int num_db = 0;
    fp.open(training_data_path.c_str(), std::ios::in|std::ios::binary);
    fp.read((char*)&num_db, sizeof(int));  // NOLINT

    std::cout << "db num: " << num_db << std::endl;

    float *xb = new float[d * num_db];

    int count = 0;
    std::vector<std::string> dbIds;

    float *feat = new float[d];
    float *xtest = new float[d];
    uint8_t *bytes = new uint8_t[d];
    float *xtest_decode = new float[d];

    float loss = 0;

    double encode_time = 0.0;
    double decode_time = 0.0;
    for (int i = 0; i < num_db; i++) {
        int idSize = 0;
        fp.read((char*)&idSize, sizeof(int));  // NOLINT
        char idName[1024] = {""};
        fp.read(idName, idSize);
        std::string idStr(idName);

        int dim_feat = 0;
        fp.read((char*)&dim_feat, sizeof(int));  // NOLINT
        if (dim_feat != d) {
            cout << "file error";
            exit(1);
        }

        fp.read(reinterpret_cast<char *>(feat), dim_feat*sizeof(float));
        std::vector<float> tmp_feat(feat, feat + d);
        std::vector<float> tmp_feat_normed = nomalize_vector(tmp_feat);

        for (int j = 0; j < dim_feat; j++) {
            xtest[j] = tmp_feat_normed[j];
        }

        // auto start = high_resolution_clock::now();
        SQuantizer->sq.compute_codes(xtest, bytes, 1);
        // auto stop = high_resolution_clock::now();
        // auto duration = duration_cast<microseconds>(stop - start);
        // cout << "\nTime taken by function : "<< duration.count() << " microseconds";
        auto start = high_resolution_clock::now();
        // SQuantizer->sq.decode(bytes, xtest_decode, 1);
        for (int j = 0; j < dim_feat; ++j) {
          xtest_decode[j] = trained_result[j] + trained_result[j + 64]*(bytes[j] + 0.5)/255.0;
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::chrono::duration<double> diff = stop - start;
        float inner_product = 0.0;
        for (int i = 0; i < 64; i++) {
          inner_product += tmp_feat_normed[i]*xtest_decode[i];
        }
        std::cout << "decode " << i << " sample: " << " time taken : "
            << duration.count() << " microseconds" << " inner product: " << inner_product << std::endl;
    }

    std::cout << "average loss: " << loss/(1.0*num_db) << std::endl;
    std::cout << "average encode time: " << encode_time/(1.0*num_db) << std::endl;
    std::cout << "average decode time: " << decode_time/(1.0*num_db) << std::endl;

    delete [] feat;
    delete [] xtest_decode;
    delete [] bytes;
    delete [] xtest;

    return 0;
}
