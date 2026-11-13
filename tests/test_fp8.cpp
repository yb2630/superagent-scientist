#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "test_harness.hpp"
#include "tilecpp/fp8/e4m3.hpp"

namespace {

using tilecpp::e4m3fn;

[[nodiscard]] float from_bits(std::uint32_t bits) {
  return std::bit_cast<float>(bits);
}

[[nodiscard]] std::uint32_t to_bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

TILECPP_TEST(fp8_decode_known_values) {
  CHECK(to_bits(e4m3fn::decode(0x00)) == 0x00000000U);
  CHECK(to_bits(e4m3fn::decode(0x80)) == 0x80000000U);
  CHECK(to_bits(e4m3fn::decode(0x01)) == to_bits(from_bits(0x3B000000U)));
  CHECK(to_bits(e4m3fn::decode(0x08)) == to_bits(from_bits(0x3C800000U)));
  CHECK(to_bits(e4m3fn::decode(0x38)) == to_bits(1.0F));
  CHECK(to_bits(e4m3fn::decode(0x7E)) == to_bits(448.0F));
  CHECK(to_bits(e4m3fn::decode(0xFE)) == to_bits(-448.0F));
  CHECK(std::isnan(e4m3fn::decode(0x7F)));
  CHECK(std::isnan(e4m3fn::decode(0xFF)));
}

TILECPP_TEST(fp8_special_values) {
  CHECK(e4m3fn::encode(0.0F) == 0x00U);
  CHECK(e4m3fn::encode(-0.0F) == 0x80U);
  CHECK(e4m3fn::encode(std::numeric_limits<float>::infinity()) ==
        e4m3fn::canonical_nan);
  CHECK(e4m3fn::encode(-std::numeric_limits<float>::infinity()) ==
        e4m3fn::canonical_nan);
  CHECK(e4m3fn::encode(std::numeric_limits<float>::quiet_NaN()) ==
        e4m3fn::canonical_nan);
  CHECK(e4m3fn::encode(-std::numeric_limits<float>::quiet_NaN()) ==
        e4m3fn::canonical_nan);

  CHECK(e4m3fn::encode(448.0F) == 0x7EU);
  CHECK(e4m3fn::encode(-448.0F) == 0xFEU);
  CHECK(e4m3fn::encode(448.5F) == 0x7EU);  // saturation
  CHECK(e4m3fn::encode(464.0F) == 0x7EU);
  CHECK(e4m3fn::encode(1.0e30F) == 0x7EU);
  CHECK(e4m3fn::encode(-1.0e30F) == 0xFEU);

  CHECK(e4m3fn::is_nan_encoding(0x7F));
  CHECK(e4m3fn::is_nan_encoding(0xFF));
  CHECK(!e4m3fn::is_nan_encoding(0x7E));
  CHECK(e4m3fn::is_finite_encoding(0x00));
  CHECK(!e4m3fn::is_finite_encoding(0x7F));
}

TILECPP_TEST(fp8_rounding_ties_to_even) {
  // 1.0 = 0x38, 1.125 = 0x39, 1.25 = 0x3A.
  CHECK(e4m3fn::encode(1.0625F) == 0x38U);  // tie -> even code 0x38
  CHECK(e4m3fn::encode(1.1875F) == 0x3AU);  // tie -> even code 0x3A
  CHECK(e4m3fn::encode(std::nextafter(1.0625F, 0.0F)) == 0x38U);
  CHECK(e4m3fn::encode(std::nextafter(1.0625F, 2.0F)) == 0x39U);
  CHECK(e4m3fn::encode(std::nextafter(1.1875F, 1.0F)) == 0x39U);
  CHECK(e4m3fn::encode(std::nextafter(1.1875F, 2.0F)) == 0x3AU);
}

TILECPP_TEST(fp8_subnormal_rounding) {
  const float min_subnormal = std::ldexp(1.0F, -9);
  const float half_min_subnormal = std::ldexp(1.0F, -10);

  CHECK(e4m3fn::encode(0.0F) == 0x00U);
  CHECK(e4m3fn::encode(half_min_subnormal) == 0x00U);  // tie -> even code 0
  CHECK(e4m3fn::encode(std::nextafter(half_min_subnormal, 1.0F)) == 0x01U);
  CHECK(e4m3fn::encode(std::nextafter(half_min_subnormal, 0.0F)) == 0x00U);
  CHECK(e4m3fn::encode(min_subnormal) == 0x01U);
  CHECK(e4m3fn::encode(1.5F * min_subnormal) == 0x02U);  // tie -> even code 2
  CHECK(e4m3fn::encode(std::nextafter(-half_min_subnormal, -1.0F)) == 0x81U);
  CHECK(e4m3fn::encode(7.0F * min_subnormal) == 0x07U);
  CHECK(e4m3fn::encode(8.0F * min_subnormal) == 0x08U);  // smallest normal
}

TILECPP_TEST(fp8_roundtrip_of_representable_values) {
  for (int code = 0; code <= 0x7E; ++code) {
    const auto byte = static_cast<std::uint8_t>(code);
    const float value = e4m3fn::decode(byte);
    CHECK_MSG(e4m3fn::encode(value) == byte, "roundtrip failed for code ",
              code);
    const auto negative_byte = static_cast<std::uint8_t>(0x80 | code);
    CHECK_MSG(e4m3fn::encode(-value) == negative_byte,
              "negative roundtrip failed for code ", code);
  }
  CHECK(e4m3fn::encode(e4m3fn::decode(0x7F)) == e4m3fn::canonical_nan);
  CHECK(e4m3fn::encode(e4m3fn::decode(0xFF)) == e4m3fn::canonical_nan);
}

TILECPP_TEST(fp8_error_detection_helpers) {
  // Only 0x7F/0xFF are NaN encodings; every other byte is finite.
  int nan_count = 0;
  for (int code = 0; code < 256; ++code) {
    if (e4m3fn::is_nan_encoding(static_cast<std::uint8_t>(code))) {
      ++nan_count;
    }
  }
  CHECK(nan_count == 2);
}

}  // namespace
