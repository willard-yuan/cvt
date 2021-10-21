#include "fasttext_model.h"

namespace cvtk {
namespace nlp {
bool FasttextModel::Init(const std::string& path) {
  fasttext_.loadModel(path);
  return true;
}
void FasttextModel::GetEmbedd(const std::string &query,
                                std::vector<float>* features) {
  fasttext::Vector vec(fasttext_.getDimension());
  fasttext_.getWordVector(vec, query);
  for (int j = 0; j < vec.size(); ++j) {
    features[j] = vec[j];
  }
  L2Norm(features, vec.size());
}
void FasttextModel::GetEmbedd(const std::string &query, float* features) {
  fasttext::Vector vec(fasttext_.getDimension());
  fasttext_.getWordVector(vec, query);
  for (int j = 0; j < vec.size(); ++j) {
    features[j] = vec[j];
  }
  L2Norm(features, vec.size());
}

void FasttextModel::L2Norm(float* vector, int d) {
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
}  // namespace nlp
}  // namespace cvtk
