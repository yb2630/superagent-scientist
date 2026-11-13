#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

#include "test_harness.hpp"
#include "tilecpp/tilecpp.hpp"

namespace {

using tilecpp::e4m3fn;
using tilecpp::error_code;
using tilecpp::matrix_view;
using tilecpp::owning_buffer;

static_assert(tilecpp::numerical_policy<e4m3fn>);
static_assert(tilecpp::execution_policy<tilecpp::sequential>);
static_assert(
    tilecpp::scale_policy<tilecpp::per_row_scale<e4m3fn>, e4m3fn>);
static_assert(!tilecpp::numerical_policy<int>);
static_assert(!tilecpp::execution_policy<int>);

/// A broken scale policy used to exercise runtime policy validation.
template <class NumericalPolicy>
struct zero_scale_policy {
  static constexpr std::string_view name = "zero-scale";

  [[nodiscard]] static constexpr std::size_t block_count(
      std::size_t columns) noexcept {
    return columns == 0 ? 0 : 1;
  }
  [[nodiscard]] static constexpr std::size_t block_extent(
      std::size_t columns) noexcept {
    return columns;
  }
  [[nodiscard]] static float compute(std::span<const float>) noexcept {
    return 0.0F;
  }
};

/// A scale policy whose block count does not tile the columns.
template <class NumericalPolicy>
struct broken_block_policy {
  static constexpr std::string_view name = "broken-block";

  [[nodiscard]] static constexpr std::size_t block_count(
      std::size_t columns) noexcept {
    return columns == 0 ? 0 : columns + 1;
  }
  [[nodiscard]] static constexpr std::size_t block_extent(
      std::size_t columns) noexcept {
    return columns == 0 ? 0 : 1;
  }
  [[nodiscard]] static float compute(std::span<const float>) noexcept {
    return 1.0F;
  }
};

template <class T>
[[nodiscard]] matrix_view<T> make_view(std::span<T> storage,
                                       std::size_t rows,
                                       std::size_t columns,
                                       std::size_t stride) {
  return CHECKED(matrix_view<T>::create(storage, rows, columns, stride)).value();
}

[[nodiscard]] std::uint32_t to_bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

TILECPP_TEST(quantize_basic_per_row) {
  owning_buffer<float> input_storage(4);
  input_storage[0] = 1.0F;
  input_storage[1] = -2.0F;
  input_storage[2] = 0.0F;
  input_storage[3] = 0.5F;
  const auto input = make_view<const float>(input_storage.span(), 2, 2, 2);

  owning_buffer<std::uint8_t> output_storage(4);
  const auto output =
      make_view<std::uint8_t>(output_storage.span(), 2, 2, 2);

  owning_buffer<float> scales(2);
  CHECKED(tilecpp::quantize_rows(input, output, scales.span()));

  // Row 0: scale = 2/448; 1/(2/448) = 224 -> 0x76, -2/(2/448) = -448 -> 0xFE.
  // Row 1 has max_abs = 0.5: 0 -> 0x00, 0.5/(0.5/448) = 448 -> 0x7E.
  CHECK(output_storage[0] == 0x76U);
  CHECK(output_storage[1] == 0xFEU);
  CHECK(output_storage[2] == 0x00U);
  CHECK(output_storage[3] == 0x7EU);
  CHECK(scales[0] == static_cast<float>(2.0 / 448.0));
  CHECK(scales[1] == static_cast<float>(0.5 / 448.0));
}

TILECPP_TEST(quantize_per_row_scale) {
  owning_buffer<float> input_storage(3);
  input_storage[0] = -4.0F;
  input_storage[1] = 2.0F;
  input_storage[2] = 0.0F;
  const auto input = make_view<const float>(input_storage.span(), 1, 3, 3);

  owning_buffer<std::uint8_t> output_storage(3);
  const auto output =
      make_view<std::uint8_t>(output_storage.span(), 1, 3, 3);
  owning_buffer<float> scales(1);
  CHECKED(tilecpp::quantize_rows(input, output, scales.span()));
  CHECK(scales[0] == static_cast<float>(4.0 / 448.0));
  CHECK(output_storage[0] == 0xFEU);  // -448
  CHECK(output_storage[1] == 0x76U);  // 224
  CHECK(output_storage[2] == 0x00U);
}

TILECPP_TEST(quantize_padded_strides_ignore_padding) {
  constexpr std::size_t rows = 2;
  constexpr std::size_t columns = 3;
  constexpr std::size_t input_stride = 5;
  constexpr std::size_t output_stride = 6;

  owning_buffer<float> input_storage(rows * input_stride);
  const float input_values[rows * input_stride] = {
      1.0F, -2.0F, 0.5F, 1.0e20F, 1.0e20F,
      0.25F, 0.125F, -2.0F, 1.0e20F, 1.0e20F,
  };
  for (std::size_t index = 0; index < input_storage.size(); ++index) {
    input_storage[index] = input_values[index];
  }
  const auto input =
      make_view<const float>(input_storage.span(), rows, columns, input_stride);

  owning_buffer<std::uint8_t> output_storage(rows * output_stride);
  for (auto& byte : output_storage.span()) {
    byte = 0xA5U;
  }
  const auto output = make_view<std::uint8_t>(output_storage.span(), rows,
                                              columns, output_stride);

  owning_buffer<float> scales(rows);
  CHECKED(tilecpp::quantize_rows(input, output, scales.span()));

  // Padding is huge; if it influenced the scale, these would not match.
  CHECK(scales[0] == static_cast<float>(2.0 / 448.0));
  CHECK(scales[1] == static_cast<float>(2.0 / 448.0));

  const std::uint8_t expected[rows * columns] = {
      0x76U, 0xFEU, 0x6EU,
      0x66U, 0x5EU, 0xFEU,
  };
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      CHECK_MSG(output_storage[row * output_stride + column] ==
                    expected[row * columns + column],
                "unexpected byte at row ", row, " column ", column);
    }
    for (std::size_t column = columns; column < output_stride; ++column) {
      CHECK_MSG(output_storage[row * output_stride + column] == 0xA5U,
                "output padding was overwritten at row ", row, " column ",
                column);
    }
  }
}

TILECPP_TEST(quantize_empty_inputs) {
  const std::span<float> empty_floats;
  const std::span<std::uint8_t> empty_bytes;
  const std::span<float> no_scales;

  const auto empty_input =
      make_view<const float>(empty_floats, 0, 0, 0);
  const auto empty_output =
      make_view<std::uint8_t>(empty_bytes, 0, 0, 0);
  CHECKED(tilecpp::quantize_rows(empty_input, empty_output, no_scales));

  const auto zero_columns_input = make_view<const float>(empty_floats, 3, 0, 0);
  const auto zero_columns_output =
      make_view<std::uint8_t>(empty_bytes, 3, 0, 0);
  CHECKED(tilecpp::quantize_rows(zero_columns_input, zero_columns_output,
                                 no_scales));

  const auto zero_rows_input = make_view<const float>(empty_floats, 0, 4, 4);
  const auto zero_rows_output =
      make_view<std::uint8_t>(empty_bytes, 0, 4, 4);
  CHECKED(tilecpp::quantize_rows(zero_rows_input, zero_rows_output,
                                 no_scales));

  // Empty rows still require exactly zero scales.
  owning_buffer<float> one_scale(1);
  CHECK_ERROR(
      (tilecpp::quantize_rows(empty_input, empty_output, one_scale.span())),
      error_code::invalid_dimensions);
}

TILECPP_TEST(quantize_rejects_invalid_layouts) {
  owning_buffer<float> input_storage(6);
  const auto input = make_view<const float>(input_storage.span(), 2, 3, 3);

  owning_buffer<std::uint8_t> output_storage(8);
  const auto wrong_shape =
      make_view<std::uint8_t>(output_storage.span(), 2, 4, 4);
  owning_buffer<float> scales(2);
  CHECK_ERROR(
      (tilecpp::quantize_rows(input, wrong_shape, scales.span())),
      error_code::invalid_dimensions);

  const auto output_ok =
      make_view<std::uint8_t>(output_storage.span(), 2, 3, 3);
  CHECK_ERROR((tilecpp::quantize_rows(input, output_ok, scales.span().first(1))),
              error_code::invalid_dimensions);
}

TILECPP_TEST(quantize_detects_invalid_scale_policies) {
  owning_buffer<float> input_storage(3);
  input_storage[0] = 1.0F;
  input_storage[1] = 2.0F;
  input_storage[2] = 3.0F;
  const auto input = make_view<const float>(input_storage.span(), 1, 3, 3);

  owning_buffer<std::uint8_t> output_storage(3);
  const auto output =
      make_view<std::uint8_t>(output_storage.span(), 1, 3, 3);
  owning_buffer<float> one_scale(1);

  CHECK_ERROR(
      (tilecpp::quantize_rows<e4m3fn, zero_scale_policy<e4m3fn>>(
          input, output, one_scale.span())),
      error_code::invalid_scale);

  owning_buffer<float> unused_scales(3);
  CHECK_ERROR(
      (tilecpp::quantize_rows<e4m3fn, broken_block_policy<e4m3fn>>(
          input, output, unused_scales.span())),
      error_code::unsupported_input);
}

TILECPP_TEST(quantize_special_values) {
  owning_buffer<float> input_storage(3);
  input_storage[0] = std::numeric_limits<float>::quiet_NaN();
  input_storage[1] = std::numeric_limits<float>::infinity();
  input_storage[2] = -std::numeric_limits<float>::infinity();
  const auto input = make_view<const float>(input_storage.span(), 1, 3, 3);

  owning_buffer<std::uint8_t> output_storage(3);
  const auto output =
      make_view<std::uint8_t>(output_storage.span(), 1, 3, 3);
  owning_buffer<float> scales(1);
  CHECKED(tilecpp::quantize_rows(input, output, scales.span()));

  CHECK(scales[0] == 1.0F);  // all values non-finite: fallback scale
  CHECK(output_storage[0] == e4m3fn::canonical_nan);
  CHECK(output_storage[1] == e4m3fn::canonical_nan);
  CHECK(output_storage[2] == e4m3fn::canonical_nan);

  owning_buffer<float> mixed_storage(3);
  mixed_storage[0] = std::numeric_limits<float>::quiet_NaN();
  mixed_storage[1] = -1.0F;
  mixed_storage[2] = std::numeric_limits<float>::infinity();
  const auto mixed =
      make_view<const float>(mixed_storage.span(), 1, 3, 3);
  owning_buffer<std::uint8_t> mixed_output_storage(3);
  const auto mixed_output =
      make_view<std::uint8_t>(mixed_output_storage.span(), 1, 3, 3);
  owning_buffer<float> mixed_scale(1);
  CHECKED(tilecpp::quantize_rows(mixed, mixed_output, mixed_scale.span()));
  CHECK(mixed_scale[0] == static_cast<float>(1.0 / 448.0));
  CHECK(mixed_output_storage[0] == e4m3fn::canonical_nan);
  CHECK(mixed_output_storage[1] == 0xFEU);
  CHECK(mixed_output_storage[2] == e4m3fn::canonical_nan);
}

TILECPP_TEST(quantize_preserves_signed_zero_and_tiny_scales) {
  owning_buffer<float> zero_storage(2);
  zero_storage[0] = -0.0F;
  zero_storage[1] = 0.0F;
  const auto zeros = make_view<const float>(zero_storage.span(), 1, 2, 2);
  owning_buffer<std::uint8_t> zero_output_storage(2);
  const auto zero_output =
      make_view<std::uint8_t>(zero_output_storage.span(), 1, 2, 2);
  owning_buffer<float> zero_scale(1);
  CHECKED(tilecpp::quantize_rows(zeros, zero_output, zero_scale.span()));
  CHECK(zero_scale[0] == 1.0F);
  CHECK(zero_output_storage[0] == 0x80U);
  CHECK(zero_output_storage[1] == 0x00U);

  const float denorm = std::numeric_limits<float>::denorm_min();
  owning_buffer<float> tiny_storage(1);
  tiny_storage[0] = denorm;
  const auto tiny = make_view<const float>(tiny_storage.span(), 1, 1, 1);
  owning_buffer<std::uint8_t> tiny_output_storage(1);
  const auto tiny_output =
      make_view<std::uint8_t>(tiny_output_storage.span(), 1, 1, 1);
  owning_buffer<float> tiny_scale(1);
  CHECKED(tilecpp::quantize_rows(tiny, tiny_output, tiny_scale.span()));
  // max_abs/448 underflows to zero; the policy clamps to the smallest float.
  CHECK(tiny_scale[0] == denorm);
  CHECK(tiny_output_storage[0] == 0x38U);  // 1.0

  owning_buffer<float> dequantized(1);
  const auto tiny_bytes =
      make_view<const std::uint8_t>(tiny_output_storage.span(), 1, 1, 1);
  const auto tiny_floats =
      make_view<float>(dequantized.span(), 1, 1, 1);
  CHECKED(tilecpp::dequantize_rows(tiny_bytes, tiny_scale.span(), tiny_floats));
  CHECK(dequantized[0] == denorm);
}

TILECPP_TEST(dequantize_rows_basic) {
  owning_buffer<std::uint8_t> input_storage(4);
  input_storage[0] = 0x7EU;  // 448
  input_storage[1] = 0xFEU;  // -448
  input_storage[2] = 0x38U;  // 1.0
  input_storage[3] = 0x80U;  // -0.0
  const auto input =
      make_view<const std::uint8_t>(input_storage.span(), 2, 2, 2);

  owning_buffer<float> output_storage(4);
  const auto output =
      make_view<float>(output_storage.span(), 2, 2, 2);
  const float scales[2] = {2.0F, 0.5F};
  CHECKED(tilecpp::dequantize_rows(input, scales, output));

  CHECK(output_storage[0] == 896.0F);
  CHECK(output_storage[1] == -896.0F);
  CHECK(output_storage[2] == 0.5F);
  CHECK(to_bits(output_storage[3]) == 0x80000000U);

  owning_buffer<std::uint8_t> nan_input_storage(1);
  nan_input_storage[0] = e4m3fn::canonical_nan;
  const auto nan_input =
      make_view<const std::uint8_t>(nan_input_storage.span(), 1, 1, 1);
  owning_buffer<float> nan_output_storage(1);
  const auto nan_output =
      make_view<float>(nan_output_storage.span(), 1, 1, 1);
  const float unit_scale[1] = {1.0F};
  CHECKED(tilecpp::dequantize_rows(nan_input, unit_scale, nan_output));
  CHECK(std::isnan(nan_output_storage[0]));
}

TILECPP_TEST(dequantize_rejects_invalid_scales_and_layouts) {
  owning_buffer<std::uint8_t> input_storage(2);
  input_storage[0] = 0x38U;
  input_storage[1] = 0x80U;
  const auto input =
      make_view<const std::uint8_t>(input_storage.span(), 1, 2, 2);
  owning_buffer<float> output_storage(2);
  const auto output = make_view<float>(output_storage.span(), 1, 2, 2);

  const float invalid_scales[] = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      0.0F,
      -1.0F,
  };
  for (const float scale : invalid_scales) {
    CHECK_ERROR(
        (tilecpp::dequantize_rows(input, std::span<const float>(&scale, 1),
                                  output)),
        error_code::invalid_scale);
  }

  owning_buffer<float> wrong_count(2);
  wrong_count[0] = 1.0F;
  wrong_count[1] = 1.0F;
  CHECK_ERROR(
      (tilecpp::dequantize_rows(input, wrong_count.span(), output)),
      error_code::invalid_dimensions);

  owning_buffer<float> good_scale(1);
  good_scale[0] = 1.0F;
  owning_buffer<float> wrong_output_storage(4);
  const auto wrong_output =
      make_view<float>(wrong_output_storage.span(), 2, 2, 2);
  CHECK_ERROR(
      (tilecpp::dequantize_rows(input, good_scale.span(), wrong_output)),
      error_code::invalid_dimensions);

  const std::span<std::uint8_t> empty_bytes;
  const std::span<float> empty_floats;
  const auto empty_input =
      make_view<const std::uint8_t>(empty_bytes, 0, 0, 0);
  const auto empty_output = make_view<float>(empty_floats, 0, 0, 0);
  const std::span<const float> no_scales;
  CHECKED(tilecpp::dequantize_rows(empty_input, no_scales, empty_output));
}

TILECPP_TEST(dequantize_respects_padded_strides) {
  constexpr std::size_t rows = 2;
  constexpr std::size_t columns = 2;
  constexpr std::size_t input_stride = 4;
  constexpr std::size_t output_stride = 3;

  owning_buffer<std::uint8_t> input_storage(rows * input_stride);
  const std::uint8_t bytes[rows * input_stride] = {
      0x38U, 0x80U, 0xAAU, 0xAAU,
      0x7EU, 0x38U, 0xAAU, 0xAAU,
  };
  for (std::size_t index = 0; index < input_storage.size(); ++index) {
    input_storage[index] = bytes[index];
  }
  const auto input = make_view<const std::uint8_t>(
      input_storage.span(), rows, columns, input_stride);

  owning_buffer<float> output_storage(rows * output_stride);
  for (auto& value : output_storage.span()) {
    value = -12345.0F;
  }
  const auto output = make_view<float>(output_storage.span(), rows, columns,
                                       output_stride);
  const float scales[rows] = {1.0F, 2.0F};
  CHECKED(tilecpp::dequantize_rows(input, scales, output));

  CHECK(output_storage[0] == 1.0F);
  CHECK(to_bits(output_storage[1]) == 0x80000000U);
  CHECK(output_storage[2] == -12345.0F);  // padding untouched
  CHECK(output_storage[3] == 896.0F);
  CHECK(output_storage[4] == 2.0F);
  CHECK(output_storage[5] == -12345.0F);
}

}  // namespace
