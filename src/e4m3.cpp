#include "tilecpp/fp8/e4m3.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace tilecpp {
namespace {

constexpr std::uint8_t sign_bit = 0x80;
constexpr std::uint8_t max_finite_encoding = 0x7E;

/// Round x to the nearest integer, ties to even.  Requires 0 <= x <= 8.
[[nodiscard]] int round_to_nearest_even(float x) noexcept {
  const float lower = std::floor(x);
  const int integer = static_cast<int>(lower);
  const float fraction = x - lower;
  if (fraction > 0.5F) {
    return integer + 1;
  }
  if (fraction < 0.5F) {
    return integer;
  }
  return (integer % 2 == 0) ? integer : integer + 1;
}

}  // namespace

std::uint8_t e4m3fn::encode(float value) noexcept {
  if (std::isnan(value)) {
    return canonical_nan;
  }
  const bool negative = std::signbit(value);
  const std::uint8_t sign = negative ? sign_bit : 0;
  const float magnitude = std::fabs(value);
  if (magnitude > max_finite) {
    // Finite overflow and +/-Infinity both reach this branch; infinity is
    // explicitly mapped to NaN below.
    return std::isinf(value) ? canonical_nan : (sign | max_finite_encoding);
  }
  if (magnitude == 0.0F) {
    return sign;
  }

  int exponent = 0;
  std::frexp(magnitude, &exponent);
  const int unbiased_exponent = exponent - 1;

  if (unbiased_exponent < 1 - exponent_bias) {
    // Subnormal range: representable magnitudes are multiples of 2^-9.
    // Rounding may land on 8, which is exactly the encoding 0x08 (smallest
    // normal), so the byte can be formed directly.
    const float scaled = std::ldexp(magnitude, 9);
    const int mantissa = round_to_nearest_even(scaled);
    return sign | static_cast<std::uint8_t>(mantissa);
  }

  // Normal range: magnitude = 1.fraction * 2^unbiased_exponent.
  const float significand = std::ldexp(magnitude, -unbiased_exponent);
  int mantissa =
      round_to_nearest_even(std::ldexp(significand - 1.0F, mantissa_bits));
  int encoded_exponent = unbiased_exponent;
  if (mantissa == 8) {
    mantissa = 0;
    ++encoded_exponent;
  }
  if (encoded_exponent > 8) {
    return sign | max_finite_encoding;
  }
  return sign | static_cast<std::uint8_t>(((encoded_exponent + exponent_bias)
                                           << mantissa_bits) |
                                          mantissa);
}

float e4m3fn::decode(std::uint8_t value) noexcept {
  const bool negative = (value & sign_bit) != 0;
  const std::uint8_t exponent = static_cast<std::uint8_t>((value >> 3) & 0x0F);
  const std::uint8_t mantissa = static_cast<std::uint8_t>(value & 0x07);
  if (exponent == 0x0F && mantissa == 0x07) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  float magnitude = 0.0F;
  if (exponent == 0) {
    magnitude = std::ldexp(static_cast<float>(mantissa), -9);
  } else {
    magnitude = std::ldexp(
        1.0F + static_cast<float>(mantissa) * 0.125F, exponent - exponent_bias);
  }
  return negative ? -magnitude : magnitude;
}

}  // namespace tilecpp
