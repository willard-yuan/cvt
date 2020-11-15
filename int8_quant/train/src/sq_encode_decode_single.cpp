
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

#include <faiss/Index.h>
#include <faiss/index_io.h>
#include <faiss/IndexScalarQuantizer.h>

int main() {

    std::string model_path = "../model/int8_sq_model.bin";

    int dim = 64;
    float xtmp[dim] = {-0.0554737,-0.115328,0.259764,-0.107465,-0.197002,-0.17915,0.0753817,-0.0202922,-0.0508304,-0.0394114,-0.0221296,0.0558961,-0.0356634,-0.220769,0.0354653,-0.0213216,0.161869,0.000317825,-0.265494,0.116488,-0.0830846,-0.123266,0.092719,-0.139823,-0.00272171,0.00929862,-0.0328674,0.0867445,-0.203404,-0.0399156,0.114142,0.0819988,-0.126456,0.0222418,0.043105,-0.0712679,-0.0269156,-0.134474,-0.000934767,0.0937075,0.139443,0.462214,-0.00856428,0.0751993,0.0704962,-0.289772,0.0786663,-0.0626779,-0.0466378,-0.013877,0.0666665,-0.0153813,0.0657903,0.0602585,-0.123214,0.0253156,0.0707286,0.0675207,-0.205772,-0.227674,0.0494444,-0.031262,-0.0797032,0.0545618};

    faiss::Index * index_model = faiss::read_index(model_path.c_str());
    faiss::IndexScalarQuantizer * SQuantizer = reinterpret_cast<faiss::IndexScalarQuantizer*>(index_model);

    float *xtest = new float[dim];
    uint8_t *bytes = new uint8_t[dim];
    
    for (int i = 0; i < dim; i++) {
        xtest[i] = xtmp[i];
    }

    SQuantizer->sq.compute_codes(xtest, bytes, 1);

    float *xtest_decode = new float[dim];
    SQuantizer->sq.decode(bytes, xtest_decode, 1);

    std::cout << "原始值(64维) " << "int8压缩后表示 " << "解码后表示" << std::endl;
    for (int i = 0; i < 64; i++) {
        std::cout << xtest[i] << ", " << unsigned(bytes[i]) << ", " << xtest_decode[i] << std::endl;
    }

    delete [] xtest;
    delete [] bytes;
    delete [] xtest_decode;

    return 0;
}
