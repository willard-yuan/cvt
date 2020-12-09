#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>
#include <utility>
#include <stack>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace cvtk {
class MathUtil {
 public:
  static std::vector<float> L2NormVec(std::vector<float> &v) {  // NOLINT
    std::vector<float> v_norm;
    float norm_v = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    float denorm_v = std::max((double)1e-12, (double)norm_v);
    for (auto it = v.begin(); it != v.end(); it++) {
      float tmp = (*it)/denorm_v;
      v_norm.push_back(tmp);
    }
    return v_norm;
  }

  static void L2NormArray(float* vector, int d) {
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
};
}  // namespace cvtk
