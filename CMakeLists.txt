cmake_minimum_required(VERSION 3.2)
project(test C CXX)


set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")


add_executable(simple_net
  "src/main.cc"
  "src/nn.cc"
  "src/csv_reader.cc"
  "src/ops/dense_layer.cc"
)


target_include_directories(simple_net PUBLIC
  "include"
)
