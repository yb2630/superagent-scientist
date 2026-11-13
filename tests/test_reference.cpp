#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "test_harness.hpp"
#include "tilecpp/tilecpp.hpp"

namespace {

using tilecpp::e4m3fn;
using tilecpp::matrix_view;
using tilecpp::owning_buffer;

constexpr std::uint8_t output_padding_byte = 0xA5U;

[[nodiscard]] std::string data_file(const char* name) {
  return std::string(TILECPP_TEST_DATA_DIR) + "/" + name;
}

[[nodiscard]] float bits_to_float(std::uint32_t bits) {
  return std::bit_cast<float>(bits);
}

[[nodiscard]] std::uint32_t float_to_bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

[[nodiscard]] std::size_t parse_size(const std::string& token) {
  return static_cast<std::size_t>(std::stoul(token, nullptr, 10));
}

[[nodiscard]] std::uint32_t parse_hex32(const std::string& token) {
  return static_cast<std::uint32_t>(std::stoul(token, nullptr, 16));
}

[[nodiscard]] std::uint8_t parse_hex8(const std::string& token) {
  return static_cast<std::uint8_t>(std::stoul(token, nullptr, 16));
}

[[nodiscard]] float parse_float_token(const std::string& token) {
  if (token == "nan") {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return bits_to_float(parse_hex32(token));
}

[[nodiscard]] std::vector<std::string> read_non_comment_lines(
    const std::string& path) {
  std::ifstream file(path);
  CHECK_MSG(file.is_open(), "cannot open reference vector file ", path);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty() || line.front() == '#') {
      continue;
    }
    lines.push_back(line);
  }
  return lines;
}

[[nodiscard]] std::vector<std::string> tokens_of(const std::string& line) {
  std::istringstream stream(line);
  std::vector<std::string> tokens;
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

struct QuantizeVector {
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::size_t input_stride = 0;
  std::size_t output_stride = 0;
  std::vector<float> input;
  std::vector<float> scales;
  std::vector<std::uint8_t> quantized;
  std::vector<float> dequantized;
};

TILECPP_TEST(reference_decodes_all_256_patterns) {
  const auto lines = read_non_comment_lines(data_file("fp8_decode.tsv"));
  CHECK(lines.size() == 256U);
  for (std::size_t index = 0; index < lines.size(); ++index) {
    const auto tokens = tokens_of(lines[index]);
    CHECK_MSG(tokens.size() == 2U, "bad decode vector line ", index);
    const auto code = parse_hex8(tokens[0]);
    const float decoded = e4m3fn::decode(code);
    if (tokens[1] == "nan") {
      CHECK_MSG(std::isnan(decoded), "expected NaN decoding for code ",
                static_cast<int>(code));
    } else {
      CHECK_MSG(float_to_bits(decoded) == parse_hex32(tokens[1]),
                "decode mismatch for code ", static_cast<int>(code));
    }
  }
}

TILECPP_TEST(reference_encodes_rounding_and_special_cases) {
  const auto lines = read_non_comment_lines(data_file("fp8_encode.tsv"));
  CHECK(lines.size() > 1000U);
  for (std::size_t index = 0; index < lines.size(); ++index) {
    const auto tokens = tokens_of(lines[index]);
    CHECK_MSG(tokens.size() == 2U, "bad encode vector line ", index);
    const float input = bits_to_float(parse_hex32(tokens[0]));
    const auto expected = parse_hex8(tokens[1]);
    CHECK_MSG(e4m3fn::encode(input) == expected, "encode mismatch for input ",
              tokens[0], " expected ", static_cast<int>(expected), " got ",
              static_cast<int>(e4m3fn::encode(input)));
  }
}

TILECPP_TEST(reference_quantizes_rows) {
  const auto lines = read_non_comment_lines(data_file("quantize.tsv"));
  std::vector<QuantizeVector> vectors;
  std::size_t position = 0;

  const auto read_section = [&](const char* keyword) {
    CHECK_MSG(position < lines.size(), "missing section ", keyword);
    auto tokens = tokens_of(lines[position++]);
    CHECK_MSG(!tokens.empty() && tokens[0] == keyword, "expected section ",
              keyword);
    tokens.erase(tokens.begin());
    return tokens;
  };

  while (position < lines.size()) {
    const auto header = tokens_of(lines[position++]);
    CHECK_MSG(header.size() == 5U && header[0] == "case",
              "bad quantize case header at line ", position - 1);
    QuantizeVector vector;
    vector.rows = parse_size(header[1]);
    vector.columns = parse_size(header[2]);
    vector.input_stride = parse_size(header[3]);
    vector.output_stride = parse_size(header[4]);

    const auto input_tokens = read_section("in");
    CHECK(input_tokens.size() == vector.rows * vector.input_stride);
    for (const auto& token : input_tokens) {
      vector.input.push_back(parse_float_token(token));
    }

    const auto scale_tokens = read_section("scales");
    CHECK(scale_tokens.size() == vector.rows);
    for (const auto& token : scale_tokens) {
      vector.scales.push_back(parse_float_token(token));
    }

    const auto quantized_tokens = read_section("quantized");
    CHECK(quantized_tokens.size() == vector.rows * vector.columns);
    for (const auto& token : quantized_tokens) {
      vector.quantized.push_back(parse_hex8(token));
    }

    const auto dequantized_tokens = read_section("dequantized");
    CHECK(dequantized_tokens.size() == vector.rows * vector.columns);
    for (const auto& token : dequantized_tokens) {
      vector.dequantized.push_back(parse_float_token(token));
    }

    const auto end_tokens = read_section("end");
    CHECK(end_tokens.empty());
    vectors.push_back(std::move(vector));
  }
  CHECK(vectors.size() >= 10U);

  for (const auto& vector : vectors) {
    const auto rows = vector.rows;
    const auto columns = vector.columns;
    const auto in_stride = vector.input_stride;
    const auto out_stride = vector.output_stride;

    owning_buffer<float> input_storage(rows * in_stride);
    for (std::size_t index = 0; index < input_storage.size(); ++index) {
      input_storage[index] = vector.input[index];
    }
    const auto input = CHECKED(matrix_view<const float>::create(
                                  input_storage.span(), rows, columns,
                                  in_stride))
                           .value();

    owning_buffer<std::uint8_t> quantized_storage(rows * out_stride);
    for (auto& byte : quantized_storage.span()) {
      byte = output_padding_byte;
    }
    const auto quantized = CHECKED(matrix_view<std::uint8_t>::create(
                                       quantized_storage.span(), rows, columns,
                                       out_stride))
                               .value();

    owning_buffer<float> scales(rows);
    CHECKED(tilecpp::quantize_rows(input, quantized, scales.span()));

    for (std::size_t row = 0; row < rows; ++row) {
      CHECK_MSG(float_to_bits(scales[row]) ==
                    float_to_bits(vector.scales[row]),
                "scale mismatch in row ", row);
      for (std::size_t column = 0; column < columns; ++column) {
        CHECK_MSG(quantized_storage[row * out_stride + column] ==
                      vector.quantized[row * columns + column],
                  "quantized byte mismatch at row ", row, " column ", column);
      }
      for (std::size_t column = columns; column < out_stride; ++column) {
        CHECK_MSG(quantized_storage[row * out_stride + column] ==
                      output_padding_byte,
                  "quantization overwrote output padding at row ", row,
                  " column ", column);
      }
    }

    owning_buffer<float> restored_storage(rows * out_stride);
    for (auto& value : restored_storage.span()) {
      value = -12345.0F;
    }
    const auto restored = CHECKED(matrix_view<float>::create(
                                      restored_storage.span(), rows, columns,
                                      out_stride))
                              .value();
    CHECKED(tilecpp::dequantize_rows(quantized, scales.span(), restored));

    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t column = 0; column < columns; ++column) {
        const float actual = restored_storage[row * out_stride + column];
        const float expected = vector.dequantized[row * columns + column];
        if (std::isnan(expected)) {
          CHECK_MSG(std::isnan(actual),
                    "expected NaN dequantized value at row ", row, " column ",
                    column);
        } else {
          CHECK_MSG(float_to_bits(actual) == float_to_bits(expected),
                    "dequantized mismatch at row ", row, " column ", column);
        }
      }
      for (std::size_t column = columns; column < out_stride; ++column) {
        CHECK_MSG(restored_storage[row * out_stride + column] == -12345.0F,
                  "dequantization overwrote output padding at row ", row,
                  " column ", column);
      }
    }
  }
}

}  // namespace
