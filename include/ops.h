#ifndef MY_OPS_H
#define MY_OPS_H

#include <cmath>
#include <vector>
#include <immintrin.h>
#include <avx2intrin.h>

#include "nn.h"
#include "utils.h"


namespace nn {
namespace ops {


class Layer {};

class DenseLayer : public Layer {
public:
  template<class T>
  Tensor<T> SimpleForword(const Tensor<T>& data, const Tensor<T>& weight, const Tensor<T>& bias) {
    CHECK_EQ(data.Rank(), 2) << "only supports 2D dense" << std::endl;
    CHECK_EQ(data.shape[1], weight.shape[1]) << "data vs. weight: dim mismatch";

    size_t batch = data.shape[0];
    size_t in_dim = data.shape[1];
    size_t out_dim = weight.shape[0];

    Tensor<T> dense_out({batch, out_dim});
    for (size_t i = 0; i < batch; ++i) {
      for (size_t j = 0; j < out_dim; ++j) {
        dense_out[i * out_dim + j] = 0;
        for (size_t k = 0; k < in_dim; ++k) {
          dense_out[i * out_dim + j] += data[i * in_dim + k] * weight[j * in_dim + k];
        }
      }
    }

    if (!bias.IsEmpty()) {
        CHECK_EQ(weight.shape[0], bias.shape[0]) << "weight vs. bias dim mismatch";
        for (size_t i = 0; i < batch; ++i) {
          for (size_t j = 0; j < out_dim; ++j) {
            dense_out[i * out_dim + j] += bias[j];
          }
        }
    }

    return std::move(dense_out);
  }

  template<class T>
  Tensor<T> OptForword(const Tensor<T>& data, const Tensor<T>& weight, const Tensor<T>& bias) {
    CHECK_EQ(data.Rank(), 2) << "only supports 2D dense" << std::endl;
    CHECK_EQ(data.shape[1], weight.shape[1]) << "data vs. weight: dim mismatch";

    size_t batch = data.shape[0];
    size_t in_dim = data.shape[1];
    size_t out_dim = weight.shape[0];

    Tensor<T> dense_out({batch, out_dim});

    size_t factor = 32;
    CHECK_EQ(in_dim % factor, 0);

    for (int32_t i = 0; i < batch; ++i) {
      // TODO: parallel
      for (int32_t j = 0; j < out_dim; ++j) {
        dense_out[i * out_dim + j] = 0;
      }
      // TODO: parallel
      for (int32_t j = 0; j < out_dim; ++j) {
        // TODO: avx2
        for (int32_t ko = 0; ko < in_dim / factor; ++ko) {
          for (int32_t k_inner = 0; k_inner < factor; ++k_inner) {
            dense_out[i * out_dim + j] += data[i * in_dim + ko * factor + k_inner] * weight[j * in_dim + ko * factor + k_inner];
          }
        }
      }
    }
    if (!bias.IsEmpty()) {
      CHECK_EQ(weight.shape[0], bias.shape[0]) << "weight vs. bias dim mismatch";
      for (int32_t i = 0; i < batch; ++i) {
        for (int32_t j = 0; j < out_dim; ++j) {
          dense_out[i * out_dim + j] += bias[j];
        }
      }
    }

    return std::move(dense_out);
  }
};


class SoftmaxLayer : public Layer {
public:
  template<class T>
  Tensor<T> SimpleForword(const Tensor<T>& data) {
    CHECK_EQ(data.shape.size(), 2) << "softmax only supports 2D";
    size_t batch = data.shape[0];
    size_t out_dim = data.shape[1];

    Tensor<T> softmax_norm(data.shape);
    for (size_t i = 0; i < batch; ++i) {
      T maxelem = std::numeric_limits<T>::min();
      for (size_t j = 0; j < out_dim; ++j) {
        T temp = data[i * out_dim + j];
        if (temp > maxelem) {
          maxelem = temp;
        }
      }
      for (size_t j = 0; j < out_dim; ++j) {
        softmax_norm[i * out_dim + j] = exp(data[i * out_dim + j] - maxelem);
      }
      T sum = 0;
      for (size_t j = 0; j < out_dim; ++j) {
        sum += softmax_norm[i * out_dim + j];
      }
      for (size_t j = 0; j < out_dim; ++j) {
        softmax_norm[i * out_dim + j] = softmax_norm[i * out_dim + j] / sum;
      }
    }

    return softmax_norm;
  }
};


}  // end of namespace nn
}  // end of namespace op

#endif  // MY_OPS_H