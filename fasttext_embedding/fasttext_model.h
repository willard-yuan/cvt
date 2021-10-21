#pragma once
#include <memory>
#include <string>
#include <vector>
#include "fasttext.h"

namespace cvtk {
namespace nlp {
class FasttextModel {
 public:
  bool Init(const std::string& path);
  void GetEmbedd(const std::string &query, float* features);
  void GetEmbedd(const std::string &query, std::vector<float>* features);
  void NormL2(const std::vector<float>& vecs,  std::vector<float>* norm);
 private:
  fasttext::FastText fasttext_;
};

}  // namespace nlp
}  // namespace cvtk
