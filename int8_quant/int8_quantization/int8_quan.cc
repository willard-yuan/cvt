#include "se/txt2vid_se/util/int8_quantization/int8_quan.h"

namespace mmu {
namespace search {

Int8Quan::Int8Quan(const std::string &model_path) {
  SQuantizer = reinterpret_cast<faiss::IndexScalarQuantizer*>(faiss::read_index(model_path.c_str()));
}

void Int8Quan::L2NomalizeVector(float* vector, int d) {
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

int Int8Quan::Int8Encode(float *x, uint8_t *bytes, size_t n_dims) {
  if (n_dims % SQuantizer->sq.code_size == 0) {
    size_t n = n_dims / SQuantizer->sq.code_size;
    for (size_t i = 0; i < n; ++i) {
      L2NomalizeVector(x + i*SQuantizer->sq.code_size, SQuantizer->sq.code_size);
    }
    SQuantizer->sq.compute_codes(x, bytes, n);
    return 1;
  }
  return 0;
}

int Int8Quan::Int8Decode(uint8_t *bytes, float *x, size_t n_dims) {
  if (n_dims % SQuantizer->sq.code_size == 0) {
    size_t n = n_dims / SQuantizer->sq.code_size;
    SQuantizer->sq.decode(bytes, x, n);
    return 1;
  }
  return 0;
}

int Int8Quan::Int8Decode(std::string &embeddding, float *x) {
  std::vector<uint8_t> codes(embeddding.begin(), embeddding.end());
  size_t n_dims = codes.size();
  uint8_t *bytes = reinterpret_cast<uint8_t*>(&codes[0]);
  if (n_dims % SQuantizer->sq.code_size == 0) {
    size_t n = n_dims / SQuantizer->sq.code_size;
    SQuantizer->sq.decode(bytes, x, n);
    return 1;
  }
  return 0;
}

int Int8Quan::Int8Decode(PhotoProfile *photo_profile, float *x) {
  std::string embeddding = photo_profile->image_embedding().cross_64d_image_embedding();
  std::vector<uint8_t> codes(embeddding.begin(), embeddding.end());
  size_t n_dims = codes.size();
  uint8_t *bytes = reinterpret_cast<uint8_t*>(&codes[0]);
  if (n_dims % SQuantizer->sq.code_size == 0) {
    size_t n = n_dims / SQuantizer->sq.code_size;
    SQuantizer->sq.decode(bytes, x, n);
    return 1;
  }
  return 0;
}

}  // namespace search
}  // namespace mmu
