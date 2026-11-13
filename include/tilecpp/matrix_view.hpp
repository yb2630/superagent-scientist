#pragma once

/// \file
/// Non-owning row-major matrix view.
///
/// A view couples a pointer to caller-owned storage with (rows, columns,
/// stride).  The stride may exceed `columns`, which models padded rows without
/// changing the logical shape: padding elements are never read or written by
/// TileCpp algorithms.  All invariants are checked once, in `create`, and the
/// resulting view is cheap to copy and pass around.
///
/// The view is deliberately not templated on a layout policy yet.  A layout
/// policy parameter can be added when column-major or tiled layouts arrive;
/// milestone 1 standardizes on contiguous, possibly padded, row-major storage.

#include <concepts>
#include <cstddef>
#include <expected>
#include <limits>
#include <span>
#include <string>
#include <type_traits>

#include "tilecpp/error.hpp"

namespace tilecpp {

/// Non-owning, validated row-major view over `rows x columns` elements of `T`.
///
/// `T` may be const-qualified (`matrix_view<const float>` for read-only input).
template <class T>
class matrix_view {
 public:
  using element_type = T;
  using value_type = std::remove_const_t<T>;

  matrix_view(const matrix_view&) noexcept = default;
  matrix_view(matrix_view&&) noexcept = default;
  matrix_view& operator=(const matrix_view&) noexcept = default;
  matrix_view& operator=(matrix_view&&) noexcept = default;

  /// Adds constness when converting, e.g. `matrix_view<float>` to
  /// `matrix_view<const float>`; the reverse conversion is rejected.
  template <class U>
    requires std::convertible_to<U*, T*>
  matrix_view(const matrix_view<U>& other) noexcept
      : data_(other.data()),
        rows_(other.rows()),
        columns_(other.columns()),
        stride_(other.stride()) {}

  /// Validates and constructs a view over `storage`.
  ///
  /// Required storage for a non-empty view is
  /// `(rows - 1) * stride + columns`; rows before the last one are fully
  /// padded, the last row only needs its logical columns.  Empty views
  /// (`rows == 0` or `columns == 0`) require no storage.
  ///
  /// Errors:
  ///  - `invalid_stride` when `stride < columns`;
  ///  - `invalid_dimensions` when the required size overflows `size_t`;
  ///  - `insufficient_storage` when `storage.size()` is too small.
  [[nodiscard]] static std::expected<matrix_view, error> create(
      std::span<T> storage, std::size_t rows, std::size_t columns,
      std::size_t stride) {
    if (stride < columns) {
      return std::unexpected(error{
          error_code::invalid_stride,
          "matrix_view: row stride " + std::to_string(stride) +
              " is smaller than column count " + std::to_string(columns)});
    }

    std::size_t required = 0;
    if (rows != 0 && columns != 0) {
      constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
      // Rows are 1-based in the formula; compute without underflow risk.
      if (rows - 1 > (max - columns) / stride) {
        return std::unexpected(error{
            error_code::invalid_dimensions,
            "matrix_view: storage requirement for shape " +
                std::to_string(rows) + "x" + std::to_string(columns) +
                " with stride " + std::to_string(stride) + " overflows size_t"});
      }
      required = (rows - 1) * stride + columns;
    }

    if (storage.size() < required) {
      return std::unexpected(error{
          error_code::insufficient_storage,
          "matrix_view: shape " + std::to_string(rows) + "x" +
              std::to_string(columns) + " with stride " + std::to_string(stride) +
              " needs " + std::to_string(required) + " elements, storage has " +
              std::to_string(storage.size())});
    }

    return matrix_view(storage.data(), rows, columns, stride);
  }

  [[nodiscard]] T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
  [[nodiscard]] std::size_t stride() const noexcept { return stride_; }
  [[nodiscard]] bool empty() const noexcept {
    return rows_ == 0 || columns_ == 0;
  }

  /// Number of logical elements (padding excluded).
  [[nodiscard]] std::size_t element_count() const noexcept {
    return rows_ * columns_;
  }

  /// Logical elements of one row; excludes padding.
  [[nodiscard]] std::span<T> row(std::size_t index) const noexcept {
    return {data_ + index * stride_, columns_};
  }

  /// Element access with row-major indexing.
  [[nodiscard]] T& operator()(std::size_t row_index,
                              std::size_t column_index) const noexcept {
    return data_[row_index * stride_ + column_index];
  }

 private:
  matrix_view(T* data, std::size_t rows, std::size_t columns,
              std::size_t stride) noexcept
      : data_(data), rows_(rows), columns_(columns), stride_(stride) {}

  T* data_;
  std::size_t rows_;
  std::size_t columns_;
  std::size_t stride_;
};

}  // namespace tilecpp
