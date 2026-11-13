#pragma once

/// \file
/// OCP FP8 E4M3FN numerical policy.
///
/// Encoding: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
/// This is the "OCP / ONNX E4M3FN" variant used by ML frameworks:
///  - normal values `1.mmm * 2^(e-7)`, subnormals `0.mmm * 2^-6 = mmm * 2^-9`;
///  - no infinities: `0x7F` and `0xFF` are the only NaN encodings;
///    `0x7E`/`0xFE` (`448`) is the largest finite magnitude;
///  - the conversion is saturating: finite values above `448` clamp to `448`.
///
/// Rounding is round-to-nearest, ties-to-even, applied to the exact value of
/// the `float` argument.  Special-value behavior:
///  - any NaN encodes to the canonical NaN pattern `0x7F`;
///  - `+Inf` and `-Inf` have no representation and encode to `0x7F`;
///  - `+0.0` and `-0.0` encode to `0x00` and `0x80` (signed zero is preserved);
///  - decoding a NaN pattern yields a quiet NaN.
///
/// The struct is a *policy*: it exposes storage type, metadata and two static
/// functions, so generic quantization code can be instantiated for other
/// formats without touching memory layout or execution code.

#include <cstdint>
#include <string_view>

namespace tilecpp {

/// OCP FP8 E4M3FN quantization policy.
struct e4m3fn {
  using storage_type = std::uint8_t;

  static constexpr std::string_view name = "e4m3fn";
  static constexpr int exponent_bits = 4;
  static constexpr int mantissa_bits = 3;
  static constexpr int exponent_bias = 7;
  static constexpr float max_finite = 448.0F;
  static constexpr float min_subnormal = 0.001953125F;  // 2^-9
  static constexpr std::uint8_t canonical_nan = 0x7F;

  /// Encodes one float: round-to-nearest-even, saturating, canonical NaN.
  [[nodiscard]] static std::uint8_t encode(float value) noexcept;

  /// Decodes one byte to the exactly representable float value (or NaN).
  [[nodiscard]] static float decode(std::uint8_t value) noexcept;

  /// True when the byte is one of the two NaN encodings (`0x7F`, `0xFF`).
  [[nodiscard]] static constexpr bool is_nan_encoding(
      std::uint8_t value) noexcept {
    return (value & 0x7FU) == 0x7FU;
  }

  /// True when the byte encodes a finite value (including signed zero).
  [[nodiscard]] static constexpr bool is_finite_encoding(
      std::uint8_t value) noexcept {
    return !is_nan_encoding(value);
  }
};

}  // namespace tilecpp
