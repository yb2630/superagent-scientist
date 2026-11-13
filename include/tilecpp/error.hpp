#pragma once

/// \file
/// Error model for runtime validation.
///
/// Every fallible operation returns `std::expected<T, tilecpp::error>`.  The
/// error carries a machine-readable code plus a human-readable message.  This
/// keeps validation failures explicit and cheap: no exceptions are used for
/// expected failure modes such as shape mismatches or insufficient storage.

#include <expected>
#include <string>
#include <string_view>
#include <utility>

namespace tilecpp {

/// Categories of runtime validation failures.
enum class error_code {
  /// Shape or dimension mismatch, including overflow in size computations.
  invalid_dimensions,
  /// Row stride smaller than the number of logical columns.
  invalid_stride,
  /// The supplied storage cannot hold the requested view.
  insufficient_storage,
  /// A scale factor is not finite and strictly positive.
  invalid_scale,
  /// A policy or configuration produced an input this build cannot support.
  unsupported_input,
};

/// Human-readable name of an error code (`"invalid_stride"`, ...).
[[nodiscard]] std::string_view to_string(error_code code) noexcept;

/// A validation failure.
struct error {
  error_code code;
  std::string message;

  error(error_code code_in, std::string message_in)
      : code(code_in), message(std::move(message_in)) {}
};

/// Formats an error as `"<code>: <message>"`.
[[nodiscard]] std::string format(const error& err);

/// Convenience alias for the result type used throughout the library.
template <class T>
using result = std::expected<T, error>;

}  // namespace tilecpp
