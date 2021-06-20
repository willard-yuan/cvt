#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "third_party/faiss/include/Index.h"
#include "third_party/faiss/include/IndexScalarQuantizer.h"
#include "third_party/faiss/include/index_io.h"

namespace mmu {
namespace search {
class Int8Quan {
 public:
  explicit Int8Quan(const std::string& model_path);                       // 单模型加载
  explicit Int8Quan(const std::string& model_conf_path, int num_source);  // 多模型加载
  ~Int8Quan();

  // 参数 n 说明：表示数组中元素的个数
  int Int8EncodeFaiss(float* x, uint8_t* bytes, size_t n_dims, bool turn_off_l2norm = false, int source = 0);
  int Int8Encode(float* x, uint8_t* bytes, size_t n_dims, bool turn_off_l2norm = false, int source = 0);

  int Int8DecodeFaiss(std::string& embeddding, float* x, int source = 0);  // NOLINT
  int Int8Decode(uint8_t* bytes, float* x, size_t n_dims, int source = 0);
  int Int8Decode(std::string& embeddding, float* x, int source = 0);  // NOLINT

  bool status();
 private:
  void L2NormalizeVector(float* v, int d);
  bool load_model_ok = true;
 private:
  std::vector<faiss::IndexScalarQuantizer*> SQuantizers_;
};
}  // namespace search
}  // namespace mmu
