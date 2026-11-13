#pragma once

/// \file
/// Scaling policies.
///
/// Numerically, quantization is `q = round(x / scale)` for a scale chosen from
/// a contiguous segment of a row.  *Which* segment and how the scale is derived
/// from it are policy decisions, separated from the conversion and memory
/// layout.  Milestone 1 ships a per-row policy; the per-block policy of
/// milestone 2 will implement the same contract.
///
/// Contract (`scale_policy` concept below):
///  - `block_count(columns)` returns the number of scales per row;
///  - `block_extent(columns)` returns the maximum segment length used to tile
///    `[0, columns)`, i.e. `block_count(columns) == ceil(columns / extent)`
///    for non-empty rows and `0` blocks for empty rows;
///  - `compute(values)` returns a finite, strictly positive scale for one
///    segment; the quantization kernel validates this at runtime;
///  - `name()` exposes a readable policy name.
///
/// Scales are stored row-major as `scales[row * block_count + block_index]`.

#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <string_view>

namespace tilecpp {

/// Compile-time contract implemented by scaling policies.
template <class Policy, class NumericalPolicy>
concept scale_policy = requires(std::span<const float> values,
                                std::size_t columns) {
  { Policy::compute(values) } -> std::convertible_to<float>;
  { Policy::block_count(columns) } -> std::convertible_to<std::size_t>;
  { Policy::block_extent(columns) } -> std::convertible_to<std::size_t>;
  { Policy::name } -> std::convertible_to<std::string_view>;
  { NumericalPolicy::max_finite } -> std::convertible_to<float>;
};

/// One scale per row: `scale = max_abs(row) / max_finite`.
///
/// `max_abs` considers only finite values and only the logical columns of the
/// row (padded stride elements are ignored).  NaN and infinities therefore do
/// not influence the scale; they are encoded as NaN by the numerical policy.
/// An all-zero, all-NaN or empty row uses `scale = 1`, so zeros encode to
/// signed zero.  If the computed scale underflows to zero (possible only for
/// rows whose largest magnitude is a denormal near the smallest float), the
/// smallest positive float is used instead.
template <class NumericalPolicy>
struct per_row_scale {
  static constexpr std::string_view name = "per-row";

  [[nodiscard]] static constexpr std::size_t block_count(
      std::size_t columns) noexcept {
    return columns == 0 ? 0 : 1;
  }

  [[nodiscard]] static constexpr std::size_t block_extent(
      std::size_t columns) noexcept {
    return columns;
  }

  [[nodiscard]] static float compute(std::span<const float> values) noexcept {
    float max_abs = 0.0F;
    for (const float value : values) {
      const float magnitude = std::fabs(value);
      if (std::isfinite(magnitude) && magnitude > max_abs) {
        max_abs = magnitude;
      }
    }
    if (max_abs == 0.0F) {
      return 1.0F;
    }
    // Double division, then one rounding to float.  The Python reference
    // mirrors this exact order to keep test vectors bit-reproducible.
    const float scale = static_cast<float>(
        static_cast<double>(max_abs) /
        static_cast<double>(NumericalPolicy::max_finite));
    if (scale == 0.0F) {
      return std::numeric_limits<float>::denorm_min();
    }
    return scale;
  }
};

}  // namespace tilecpp
