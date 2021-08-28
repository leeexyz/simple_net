#include <ostream>
#include "nn.h"


namespace nn {


template<>
std::ostream& operator<<(std::ostream&os, const Tensor<int>& t) {
  os << "Tensor(";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    os << t.shape[i];
    if (i + 1 != t.shape.size()) {
      os << ",";
    }
  }
  os << ", dtype=int32)";
  return os;
}

template<>
std::ostream& operator<<(std::ostream&os, const Tensor<float>& t) {
  os << "Tensor(";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    os << t.shape[i];
    if (i + 1 != t.shape.size()) {
      os << ",";
    }
  }
  os << ", dtype=float32)";
  return os;
}


}  // end of namespace nn