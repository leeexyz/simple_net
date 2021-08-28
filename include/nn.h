#ifndef MY_NN_H
#define MY_NN_H

#include <vector>
#include "utils.h"


namespace nn {

// sample version of ND Tensor
// data is represented as 1D vector
template<class T>
class Tensor {
public:
  Tensor() {}

  explicit Tensor(std::vector<size_t> s) : shape(std::move(s)) {
    size_t dim = 1;
    for (auto i : shape) { dim *= i; }
    data.resize(dim);
  }

  explicit Tensor(std::vector<size_t> s, std::vector<T> d) : shape(std::move(s)), data(d) {
    size_t dim = 1;
    for (auto i : shape) { dim *= i; }
    CHECK_EQ(dim, data.size()) << "dim mismatch";
  }

  ~Tensor() {}

  bool IsEmpty() const { return shape.size() == 0; }

  size_t Rank () const { return shape.size(); }

  bool ShapeEqual(const Tensor& other) {
    if (shape.size() != other.shape.size()) {
        return false;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] != other.shape[i]) {
          return false;
      }
    }
    return true;
  }

  const T operator[](size_t i) const {
      if (i >= data.size()) {
        CHECK(false) << "data out of boundary";
      }
      return data[i];
  }

  T& operator[](size_t i) {
      if (i >= data.size()) {
        CHECK(false) << "data out of boundary";
      }
      return data[i];
  }

  std::vector<T> data;
  std::vector<size_t> shape;
  // not used now
  std::string dtype{"float32"};
};


template<class T>
std::ostream& operator<<(std::ostream&os, const Tensor<T>& t);

}  // end of namespace nn

#endif  // MY_NN_H