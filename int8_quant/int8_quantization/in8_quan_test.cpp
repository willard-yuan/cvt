#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

#include "se/txt2vid_se/util/int8_quantization/int8_quan.h"

int main() {
    std::string model_path = "int8_sq_model.bin";

    float xtmp[64] = {0.015064209,0.3062325,0.024975155,-0.027624095,-0.018168708,0.3550972,0.18380141,-0.15299813,-0.003499319,-0.028220521,0.23942424,-0.046249546,-0.12517187,-0.050695702,-0.07434918,-0.051789954,0.05921653,-0.0015975884,0.18285228,-0.023536965,-0.030154422,0.21079092,-0.11523904,0.05491502,0.014730556,0.20212944,-0.17337869,0.22581457,0.008422376,-0.09547719,0.03787558,0.051486474,0.09308339,0.052380774,-0.11707104,0.11351845,-0.018889287,0.16525868,-0.010684474,0.013779431,-0.08218948,-0.20487122,-0.16571212,0.002390551,-0.13688043,0.045602906,-0.18389857,-0.119626194,0.074221164,0.011198597,-0.0074417717,0.2222876,-0.03175717,0.088812634,0.08842016,-0.06632991,-0.1756601,0.083846755,-0.052951403,0.014542198,-0.21969591,0.0074771945,-0.07975674,0.1297228};

    mmu::search::Int8Quan Int8QuanModel = mmu::search::Int8Quan(model_path);
    float *xtest = new float[64];
    uint8_t *bytes = new uint8_t[64];

    for (int i = 0; i < 64; i++) {
        xtest[i] = xtmp[i];
    }

    int status = Int8QuanModel.Int8Encode(xtest, bytes, 64);

    float *xtest_decode = new float[64];
    status = Int8QuanModel.Int8Decode(bytes, xtest_decode, 64);

    std::cout << "原始值(64维) " << "int8压缩后表示 " << "解码后表示" << std::endl;
    float inner_product = 0.0;
    for (int i = 0; i < 64; i++) {
        std::cout << xtest[i] << ", " << unsigned(bytes[i]) << ", " << xtest_decode[i] << std::endl;
        inner_product += xtest[i]*xtest_decode[i];
    }

    std::cout << "inner_product: " << inner_product << std::endl;

    delete [] xtest;
    delete [] bytes;
    delete [] xtest_decode;

    return 0;
}
