#pragma once

/// \file
/// Generic scalar quantization kernels.
///
/// Three independent axes meet here:
///  - *numerical* (`NumericalPolicy`, e.g. `e4m3fn`): value encoding/decoding;
///  - *layout* (`matrix_view`): owning storage is separate, rows may be padded;
///  - *execution* (`ExecutionPolicy`, e.g. `sequential`): who runs the rows.
/// The scaling policy decides how one scale is derived from a row segment.
///
/// Both kernels are row-parallel by construction: every row has disjoint
/// output bytes, scale slots and output padding.  That property is what makes
/// the same code usable by the later thread-pool execution policy.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tilecpp/error.hpp"
#include "tilecpp/execution/sequential.hpp"
#include "tilecpp/fp8/e4m3.hpp"
#include "tilecpp/matrix_view.hpp"
#include "tilecpp/quantization/scale_policy.hpp"

namespace tilecpp {

/// Compile-time contract for numerical policies.
template <class Policy>
concept numerical_policy =
    requires(float value, typename Policy::storage_type encoded) {
      typename Policy::storage_type;
      { Policy::encode(value) } -> std::same_as<typename Policy::storage_type>;
      { Policy::decode(encoded) } -> std::same_as<float>;
      { Policy::max_finite } -> std::convertible_to<float>;
      { Policy::name } -> std::convertible_to<std::string_view>;
    };

/// Compile-time contract for execution policies (see sequential.hpp).
template <class Policy>
concept execution_policy =
    requires(const Policy& policy, std::size_t rows,
             void (*function)(std::size_t)) {
      policy.for_each_row(rows, function);
    };

namespace detail {

[[nodiscard]] inline std::expected<void, error> validate_matching_shapes(
    std::size_t input_rows, std::size_t input_columns, std::size_t output_rows,
    std::size_t output_columns, std::string_view operation) {
  if (input_rows != output_rows || input_columns != output_columns) {
    return std::unexpected(error{
        error_code::invalid_dimensions,
        std::string(operation) + ": input shape " + std::to_string(input_rows) +
            "x" + std::to_string(input_columns) + " differs from output shape " +
            std::to_string(output_rows) + "x" +
            std::to_string(output_columns)});
  }
  return {};
}

/// Number of required scales, validating the scaling policy's layout.
template <class ScalePolicy>
[[nodiscard]] inline std::expected<std::size_t, error> validate_scale_layout(
    std::size_t rows, std::size_t columns, std::string_view operation) {
  const std::size_t blocks = ScalePolicy::block_count(columns);
  if (columns == 0) {
    if (blocks != 0) {
      return std::unexpected(error{
          error_code::unsupported_input,
          std::string(operation) + ": non-empty scale layout for empty rows"});
    }
    return 0;
  }

  const std::size_t extent = ScalePolicy::block_extent(columns);
  if (extent == 0) {
    return std::unexpected(error{
        error_code::unsupported_input,
        std::string(operation) + ": scale policy returned zero block extent"});
  }
  const std::size_t expected_blocks = 1 + (columns - 1) / extent;
  if (blocks != expected_blocks) {
    return std::unexpected(error{
        error_code::unsupported_input,
        std::string(operation) + ": scale policy block count " +
            std::to_string(blocks) + " does not tile " +
            std::to_string(columns) + " columns with extent " +
            std::to_string(extent)});
  }

  constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
  if (blocks != 0 && rows > max / blocks) {
    return std::unexpected(error{
        error_code::invalid_dimensions,
        std::string(operation) + ": scale count overflows size_t"});
  }
  return rows * blocks;
}

[[nodiscard]] inline std::expected<void, error> validate_scale_value(
    float scale, std::string_view operation) {
  if (!std::isfinite(scale) || scale <= 0.0F) {
    return std::unexpected(error{
        error_code::invalid_scale,
        std::string(operation) +
            ": scale must be finite and positive, got " +
            (std::isnan(scale) ? "NaN" : std::to_string(scale))});
  }
  return {};
}

}  // namespace detail

/// Quantizes `input` into FP8 bytes, writing one scale per row segment.
///
/// `scales.size()` must equal `rows * ScalePolicy::block_count(columns)`
/// (for the default per-row policy: one scale per row).  Input and output must
/// have identical logical shapes; either may be padded (stride > columns), in
/// which case padding is neither read nor written.  Empty inputs succeed
/// without touching any output.
///
/// Scale factors are produced by `ScalePolicy` and every scale is validated.
/// The digitization is round-to-nearest-even with saturation; see the
/// numerical policy for special values.
///
/// \returns `std::unexpected(error)` on shape, size or scale violations.
template <class NumericalPolicy = e4m3fn,
          class ScalePolicy = per_row_scale<NumericalPolicy>,
          class ExecutionPolicy = sequential>
  requires numerical_policy<NumericalPolicy> &&
           scale_policy<ScalePolicy, NumericalPolicy> &&
           execution_policy<ExecutionPolicy>
[[nodiscard]] std::expected<void, error> quantize_rows(
    matrix_view<const float> input,
    matrix_view<typename NumericalPolicy::storage_type> output,
    std::span<float> scales, ExecutionPolicy execution = {}) {
  auto shapes = detail::validate_matching_shapes(
      input.rows(), input.columns(), output.rows(), output.columns(),
      "quantize_rows");
  if (!shapes) {
    return shapes;
  }

  auto required_scales = detail::validate_scale_layout<ScalePolicy>(
      input.rows(), input.columns(), "quantize_rows");
  if (!required_scales) {
    return std::unexpected(std::move(required_scales.error()));
  }
  if (scales.size() != *required_scales) {
    return std::unexpected(error{
        error_code::invalid_dimensions,
        "quantize_rows: scale buffer has " + std::to_string(scales.size()) +
            " entries, expected " + std::to_string(*required_scales)});
  }
  if (input.empty()) {
    return {};
  }

  const std::size_t columns = input.columns();
  const std::size_t blocks = ScalePolicy::block_count(columns);
  const std::size_t extent = ScalePolicy::block_extent(columns);
  // One error slot per row: distinct elements may be written concurrently by a
  // future parallel execution policy, and the first row error is reported.
  std::vector<std::optional<error>> failures(input.rows());

  execution.for_each_row(input.rows(), [&](std::size_t row) {
    const std::span<const float> source = input.row(row);
    const std::span<typename NumericalPolicy::storage_type> destination =
        output.row(row);
    for (std::size_t block = 0; block < blocks; ++block) {
      const std::size_t begin = block * extent;
      const std::size_t length = std::min(extent, columns - begin);
      const float scale = ScalePolicy::compute(source.subspan(begin, length));
      auto valid = detail::validate_scale_value(scale, "quantize_rows");
      if (!valid) {
        failures[row] = std::move(valid.error());
        return;
      }
      scales[row * blocks + block] = scale;
      for (std::size_t index = 0; index < length; ++index) {
        const float value = source[begin + index];
        // Double division mirrors the Python reference; the cast to float makes
        // the single-precision rounding point explicit before E4M3 rounding.
        const float normalized = static_cast<float>(
            static_cast<double>(value) / static_cast<double>(scale));
        destination[begin + index] = NumericalPolicy::encode(normalized);
      }
    }
  });

  for (auto& failure : failures) {
    if (failure.has_value()) {
      return std::unexpected(std::move(*failure));
    }
  }
  return {};
}

/// Dequantizes FP8 bytes using the scales produced by `quantize_rows`.
///
/// Layout and scale count follow the same rules as quantization.  Every scale
/// must be finite and strictly positive.  NaN bytes produce NaN; infinities do
/// not exist in the E4M3FN encoding.
template <class NumericalPolicy = e4m3fn,
          class ScalePolicy = per_row_scale<NumericalPolicy>,
          class ExecutionPolicy = sequential>
  requires numerical_policy<NumericalPolicy> &&
           scale_policy<ScalePolicy, NumericalPolicy> &&
           execution_policy<ExecutionPolicy>
[[nodiscard]] std::expected<void, error> dequantize_rows(
    matrix_view<const typename NumericalPolicy::storage_type> input,
    std::span<const float> scales, matrix_view<float> output,
    ExecutionPolicy execution = {}) {
  auto shapes = detail::validate_matching_shapes(
      input.rows(), input.columns(), output.rows(), output.columns(),
      "dequantize_rows");
  if (!shapes) {
    return shapes;
  }

  auto required_scales = detail::validate_scale_layout<ScalePolicy>(
      input.rows(), input.columns(), "dequantize_rows");
  if (!required_scales) {
    return std::unexpected(std::move(required_scales.error()));
  }
  if (scales.size() != *required_scales) {
    return std::unexpected(error{
        error_code::invalid_dimensions,
        "dequantize_rows: scale buffer has " + std::to_string(scales.size()) +
            " entries, expected " + std::to_string(*required_scales)});
  }
  if (input.empty()) {
    return {};
  }

  const std::size_t columns = input.columns();
  const std::size_t blocks = ScalePolicy::block_count(columns);
  const std::size_t extent = ScalePolicy::block_extent(columns);
  // One error slot per row; see quantize_rows.
  std::vector<std::optional<error>> failures(input.rows());

  execution.for_each_row(input.rows(), [&](std::size_t row) {
    const std::span<const typename NumericalPolicy::storage_type> source =
        input.row(row);
    const std::span<float> destination = output.row(row);
    for (std::size_t block = 0; block < blocks; ++block) {
      const std::size_t begin = block * extent;
      const std::size_t length = std::min(extent, columns - begin);
      const float scale = scales[row * blocks + block];
      auto valid = detail::validate_scale_value(scale, "dequantize_rows");
      if (!valid) {
        failures[row] = std::move(valid.error());
        return;
      }
      for (std::size_t index = 0; index < length; ++index) {
        const float decoded = NumericalPolicy::decode(source[begin + index]);
        destination[begin + index] = static_cast<float>(
            static_cast<double>(decoded) * static_cast<double>(scale));
      }
    }
  });

  for (auto& failure : failures) {
    if (failure.has_value()) {
      return std::unexpected(std::move(*failure));
    }
  }
  return {};
}

}  // namespace tilecpp
