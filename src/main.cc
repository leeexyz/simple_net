#include <tuple>
#include <chrono>
#include <iomanip>

#include "ops.h"

using namespace nn;
using namespace utils;


template<typename T>
void Print1DTensor(const Tensor<T>& t) {
  std::cout << t << std::endl;
  for (size_t j = 0; j < t.shape[1]; ++j) {
    std::cout << t[j];
    if (j + 1 != t.shape[1]) {
      std::cout << ",";
    }
  }
  std::cout << std::endl;
}


template<typename T>
void Print2DTensor(const Tensor<T>& t) {
  std::cout << t << std::endl;
  for (size_t i = 0; i < t.shape[0]; ++i) {
    for (size_t j = 0; j < t.shape[1]; ++j) {
      std::cout << t[i * t.shape[1] + j];
      if (j + 1 != t.shape[1]) {
        std::cout << ",";
      }
    }
    std::cout << std::endl;
  }
}


auto LoadData(const std::string& fdata, const std::string& fweight) {
  auto data1 = ReadCSV(fdata);
  auto data2 = ReadCSV(fweight);

  std::vector<float> digits_data;
  std::vector<int> labels_data;
  for (size_t i = 0; i < data1.size(); ++i) {
    CHECK_EQ(data1[i].size(), 65) << "data is corruptted!";
    for (size_t j = 0; j < data1[i].size(); ++j) {
      if (j + 1 == data1[i].size()) {
        labels_data.push_back(data1[i][j]);
      } else {
        digits_data.push_back(data1[i][j]);
      }
    }
  }

  // transpose
  std::vector<std::vector<float>> data2_t(data2[0].size(), std::vector<float>());
  for (int i = 0; i < data2.size(); ++i) {
    for (int j = 0; j < data2[i].size(); ++j) {
      data2_t[j].push_back(data2[i][j]);
    }
  }
  data2 = data2_t;

  std::vector<float> weights_data;
  std::vector<float> bias_data;
  for (size_t i = 0; i < data2.size(); ++i) {
    CHECK_EQ(data1[i].size(), 65) << "weight is corruptted!";
    for (size_t j = 0; j < data2[i].size(); ++j) {
      if (j + 1 == data2[i].size()) {
        bias_data.push_back(data2[i][j]);
      } else {
        weights_data.push_back(data2[i][j]);
      }
    }
  }

  Tensor<float> digits({data1.size(), data1[0].size() - 1}, digits_data);
  Tensor<int> labels({data1.size()}, labels_data);
  Tensor<float> weights({data2.size(), data2[0].size() - 1}, weights_data);
  Tensor<float> bias({data2.size()}, bias_data);

  CHECK(digits.ShapeEqual(Tensor<float>({1797, 64})));
  CHECK(labels.ShapeEqual(Tensor<int>({1797})));
  CHECK(weights.ShapeEqual(Tensor<float>({10, 64})));
  CHECK(bias.ShapeEqual(Tensor<float>({10})));

  return std::make_tuple(digits, labels, weights, bias);
}


void PrintTimeElapsed(std::chrono::time_point<std::chrono::high_resolution_clock> t_begin, const std::string& info) {
  double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - t_begin).count();
  std::cout << "Time elapsed for " << info << ": " << std::fixed << std::setprecision(6) << duration << "s" << std::endl;
}

template<class T, class L = int>
class Network {
public:
  explicit Network(Tensor<T> digits, Tensor<L> labels, Tensor<T> weights, Tensor<T> bias) :
    digits_(std::move(digits)), labels_(labels), weights_(weights), bias_(bias) {}

  void RunNative() {
    auto t_begin = std::chrono::high_resolution_clock::now();
    // inference
    Tensor<float> dense_out = ops::DenseLayer().SimpleForword<float>(digits_, weights_, bias_);
    // Print2DTensor<float>(dense_out);
    Tensor<float> softmax_out = ops::SoftmaxLayer().SimpleForword<float>(dense_out);
    // Print2DTensor<float>(softmax_out);
    PrintTimeElapsed(t_begin, "Part 1");
    Predict(softmax_out);
  }

  void RunOpt() {
    auto t_begin = std::chrono::high_resolution_clock::now();
    // inference
    Tensor<float> dense_out = ops::DenseLayer().OptForword<float>(digits_, weights_, bias_);
    // Print2DTensor<float>(dense_out);
    Tensor<float> softmax_out = ops::SoftmaxLayer().SimpleForword<float>(dense_out);
    // Print2DTensor<float>(softmax_out);
    PrintTimeElapsed(t_begin, "Part 2");
    Predict(softmax_out);
  }

  void Predict(const Tensor<T>& softmax_out) {
    // predict
    size_t pass = 0;
    size_t batch = softmax_out.shape[0];
    size_t out_dim = softmax_out.shape[1];
    CHECK_EQ(batch, labels_.shape[0]);
    for (size_t i = 0; i < batch; ++i) {
      size_t max_index = 0;
      float max_value = 0;
      for (size_t j = 0; j < out_dim; ++j) {
        if (softmax_out[i * out_dim + j] > max_value) {
          max_index = j;
          max_value = softmax_out[i * out_dim + j];
        }
      }
      // std::cout << "Index:" << i + 1
      //           << " Pred Num:" << max_index
      //           << " Real Num:" << labels[i]
      //           << " Prob:" << max_value << std::endl;
      pass += (max_index == labels_[i]);
    }

    float result = static_cast<float>(pass) / static_cast<float>(batch) * 100;
    std::cout << "The prediction is: "
              << result << "% Pass:" << pass << " Total:" << batch << std::endl;
  }

private:
  Tensor<T> digits_;
  Tensor<L> labels_;
  Tensor<T> weights_;
  Tensor<T> bias_;
};


int main(int argc, char *argv[]) {
  if (argc != 3) {
    CHECK(false)
      << "Usage ./dense_test digits.csv weights.csv";
  }
  // load data
  std::string fdata = argv[1];
  std::string fweight = argv[2];

  Tensor<float> digits;
  Tensor<int> labels;
  Tensor<float> weights;
  Tensor<float> bias;
  std::tie(digits, labels, weights, bias) = LoadData(fdata, fweight);

  Network<float> net(digits, labels, weights, bias);
  net.RunNative();
  net.RunOpt();

  return 0;
}