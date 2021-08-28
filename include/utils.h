#ifndef MY_UTILS_H
#define MY_UTILS_H

#include <string>
#include <vector>
#include <ostream>
#include <iostream>


class Logger {
public:
  Logger(const char* file, int line) : ostream_(std::cerr) {
    ostream_ << "[" << file << ":" << line << "] ";
  }

  ~Logger() { ostream_ << '\n'; abort(); }
  std::ostream& stream() { return ostream_; }

private:
    std::ostream& ostream_;
};

#define CHECK(x)                                         \
  if (!(x))                                              \
    Logger(__FILE__, __LINE__).stream()                  \
      << "Check failed: " << #x ", "

#define CHECK_BINARY_OP(name, op, x, y)                  \
  if (!(x op y))                                         \
    Logger(__FILE__, __LINE__).stream()                  \
      << "Check failed: " << #x " " #op " " #y << ", "

#define CHECK_LT(x, y) CHECK_BINARY_OP(_LT, <, x, y)
#define CHECK_GT(x, y) CHECK_BINARY_OP(_GT, >, x, y)
#define CHECK_LE(x, y) CHECK_BINARY_OP(_LE, <=, x, y)
#define CHECK_GE(x, y) CHECK_BINARY_OP(_GE, >=, x, y)
#define CHECK_EQ(x, y) CHECK_BINARY_OP(_EQ, ==, x, y)
#define CHECK_NE(x, y) CHECK_BINARY_OP(_NE, !=, x, y)


namespace utils {

// fcsv: csv file path
// sep: seperater
// skip_header: if skip the header
std::vector<std::vector<float>> ReadCSV(std::string fcsv, std::string sep=",", bool skip_header=false);


}  // end of namespace utils

#endif  // MY_UTILS_H