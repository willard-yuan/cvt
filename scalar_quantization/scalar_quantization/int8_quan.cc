#include "int8_quan.h"

namespace mmu {
namespace search {

Int8Quan::Int8Quan(const std::string& model_path) {
  std::ifstream fin(model_path);
  if (!fin.good()) {
    std::cout << "model file is not exists" << std::endl;
    load_model_ok = false;
    return;
  }
  load_model_ok = true;
  auto SQuantizer_ = reinterpret_cast<faiss::IndexScalarQuantizer*>(faiss::read_index(model_path.c_str()));
  SQuantizers_.emplace_back(SQuantizer_);
}

Int8Quan::~Int8Quan() {}

Int8Quan::Int8Quan(const std::string& model_conf_path, int num_source) {
  std::ifstream fin(model_conf_path);
  if (!fin.good()) {
    std::cout << "model file is not exists" << std::endl;
    load_model_ok = false;
    return;
  }
  load_model_ok = true;
  nlohmann::json model_conf;
  fin >> model_conf;
  fin.close();
  SQuantizers_.clear();
  for (int i = 0; i < model_conf.size(); ++i) {
    std::string tmp_model_path = model_conf[std::to_string(i)]["model_path"];
    std::cout << "load model: " << tmp_model_path << std::endl;
    auto SQuantizer_ =
        reinterpret_cast<faiss::IndexScalarQuantizer*>(faiss::read_index(tmp_model_path.c_str()));
    SQuantizers_.emplace_back(SQuantizer_);
  }
}

// 判断模型是否加载成功，如果模型bin文件不存在，则返回值为false；反之，返回true
bool Int8Quan::status(){
  return load_model_ok;
}

void Int8Quan::L2NormalizeVector(float* vector, int d) {
  double accum = 0.0;
  for (int i = 0; i < d; ++i) {
    accum += vector[i] * vector[i];
  }
  accum = sqrt(accum);
  float denorm_v = std::max((double)1e-12, (double)accum);
  for (int i = 0; i < d; ++i) {
    vector[i] = vector[i] / denorm_v;
  }
}

int Int8Quan::Int8EncodeFaiss(float* x, uint8_t* bytes, size_t n_dims, bool turn_off_l2norm, int source) {
  if (n_dims % SQuantizers_[source]->sq.code_size == 0) {
    size_t n = n_dims / SQuantizers_[source]->sq.code_size;
    if (!turn_off_l2norm) {
      for (size_t i = 0; i < n; ++i) {
        L2NormalizeVector(x + i * SQuantizers_[source]->sq.code_size, SQuantizers_[source]->sq.code_size);
      }
    }
    SQuantizers_[source]->sq.compute_codes(x, bytes, n);
    return 1;
  }
  return 0;
}

int Int8Quan::Int8Encode(float* x, uint8_t* bytes, size_t n_dims, bool turn_off_l2norm, int source) {
  if (n_dims % SQuantizers_[source]->sq.code_size != 0) {
    return 0;
  }
  if (!turn_off_l2norm) {
    L2NormalizeVector(x, SQuantizers_[source]->sq.code_size);
  }
  for (size_t i = 0; i < SQuantizers_[source]->sq.code_size; i++) {
    float xi = 0;
    if (SQuantizers_[source]->sq.trained[i + SQuantizers_[source]->sq.code_size] != 0) {
      xi = (x[i] - SQuantizers_[source]->sq.trained[i]) /
           SQuantizers_[source]->sq.trained[i + SQuantizers_[source]->sq.code_size];
    }
    if (xi < 0) {
      xi = 0;
    }
    if (xi > 1.0) {
      xi = 1.0;
    }
    bytes[i] = (int)(255 * xi);
  }
  return 1;
}

int Int8Quan::Int8Decode(uint8_t* bytes, float* x, size_t n_dims, int source) {
  if (n_dims % SQuantizers_[source]->sq.code_size == 0) {
    size_t n = n_dims / SQuantizers_[source]->sq.code_size;
    SQuantizers_[source]->sq.decode(bytes, x, n);
    return 1;
  }
  return 0;
}

int Int8Quan::Int8DecodeFaiss(std::string& embedding, float* x, int source) {
  std::vector<uint8_t> codes(embedding.begin(), embedding.end());
  size_t n_dims = codes.size();
  uint8_t* bytes = reinterpret_cast<uint8_t*>(&codes[0]);
  if (n_dims % SQuantizers_[source]->sq.code_size == 0) {
    size_t n = n_dims / SQuantizers_[source]->sq.code_size;
    SQuantizers_[source]->sq.decode(bytes, x, n);
    return 1;
  }
  return 0;
}

int Int8Quan::Int8Decode(std::string& embedding, float* x, int source) {
  if (embedding.empty()) {
    return 0;
  }
  size_t n_dims = embedding.size();
  uint8_t* bytes = reinterpret_cast<uint8_t*>(&embedding[0]);
  if (n_dims % SQuantizers_[source]->sq.code_size != 0) {
    return 0;
  }
  for (int i = 0; i < SQuantizers_[source]->sq.code_size; ++i) {
    x[i] =
        SQuantizers_[source]->sq.trained[i] +
        SQuantizers_[source]->sq.trained[i + SQuantizers_[source]->sq.code_size] * (bytes[i] + 0.5) / 255.0;
  }
  return 1;
}

}  // namespace search
}  // namespace mmu
