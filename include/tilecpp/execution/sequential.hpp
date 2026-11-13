#pragma once

/// \file
/// Execution policies.
///
/// The quantize/dequantize kernels are written against the small
/// `for_each_row` contract below.  Milestone 1 provides only the sequential
/// policy; a thread-pool policy will implement the same contract in
/// milestone 3 without changing the numerical code.
///
/// A conforming policy must invoke the callback for every row index in
/// `[0, rows)` exactly once and return only after all invocations finished.
/// Each invocation owns a distinct output row, so rows may run concurrently.

#include <cstddef>
#include <string_view>
#include <utility>

namespace tilecpp {

/// Runs rows one after another on the calling thread.
struct sequential {
  static constexpr std::string_view name = "sequential";

  template <class Function>
  void for_each_row(std::size_t rows, Function&& function) const {
    for (std::size_t row = 0; row < rows; ++row) {
      std::forward<Function>(function)(row);
    }
  }
};

}  // namespace tilecpp
