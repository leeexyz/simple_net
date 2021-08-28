#include <sstream>
#include <fstream>

#include "utils.h"

namespace utils {

// simple versoin csv reader
std::vector<std::vector<float>> ReadCSV(std::string fcsv, std::string sep, bool skip_header) {

  std::ifstream ifs(fcsv);
  CHECK(ifs.is_open()) << "Open file " << fcsv << " failed!";

  std::vector<std::vector<float>> table;

  std::string line;
  // skip header anyway
  if (skip_header) {
    std::getline(ifs, line);
  }

  while (std::getline(ifs, line)) {
    size_t pos = 0;
    std::vector<float> vec;
    while(line.substr(pos).find(sep) != std::string::npos) {
      std::string str = line.substr(pos, line.substr(pos).find(sep));
      vec.emplace_back(std::stof(str));
      pos += line.substr(pos).find(sep) + sep.size();
    }
    vec.emplace_back(std::stof(line.substr(pos)));

    table.emplace_back(vec);
  }

  return std::move(table);
}


}  // end of namespace utils

// // for testing
// int main () {
//   auto digits = utils::ReadCSV("./digits_1.csv");
//   auto weights = utils::ReadCSV("./weights.csv");
//
//   size_t col = 0;
//   for (size_t i = 0; i < digits.size(); ++i) {
//     for (size_t j = 0; j < digits[i].size(); ++j) {
//       // if (j == 64) {
//       //   std::cout << digits[i][j] << ",";
//       // }
//     }
//     if (digits[i].size() > col) {
//       col = digits[i].size();
//     }
//   }
//   std::cout << std::endl;
//
//   CHECK_EQ(digits.size(), 1797) << " row of digits is " << digits.size() << std::endl;;
//   CHECK_EQ(col, 65) << " col of digits is " << col << std::endl;
//
//   return 0;
// }