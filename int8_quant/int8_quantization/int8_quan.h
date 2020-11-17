#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include "third_party/faiss/include/Index.h"
#include "third_party/faiss/include/index_io.h"
#include "third_party/faiss/include/IndexScalarQuantizer.h"
#include "se/txt2vid_se/profile/proto/photo_profile.pb.h"

using search::profile::PhotoProfile;
namespace mmu {
namespace search {
class Int8Quan {
 public:
     explicit Int8Quan(const std::string &model_path);
     // 参数 n 说明：表示数组中元素的个数
     int Int8Encode(float *x, uint8_t *bytes, size_t n_dims);
     int Int8Decode(uint8_t *bytes, float *x, size_t n_dims);
     int Int8DecodeFaiss(std::string &embeddding, float *x);  // NOLINT
     int Int8Decode(std::string &embeddding, float *x);  // NOLINT
     int Int8Decode(PhotoProfile *photo_profile, float *x);
 private:
     void L2NomalizeVector(float* v, int d);
 private:
     faiss::IndexScalarQuantizer *SQuantizer;
};
}  // namespace search
}  // namespace mmu
