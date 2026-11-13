#include "tilecpp/error.hpp"

#include <string>

namespace tilecpp {

std::string_view to_string(error_code code) noexcept {
  switch (code) {
    case error_code::invalid_dimensions:
      return "invalid_dimensions";
    case error_code::invalid_stride:
      return "invalid_stride";
    case error_code::insufficient_storage:
      return "insufficient_storage";
    case error_code::invalid_scale:
      return "invalid_scale";
    case error_code::unsupported_input:
      return "unsupported_input";
  }
  return "unknown";
}

std::string format(const error& err) {
  return std::string(to_string(err.code)) + ": " + err.message;
}

}  // namespace tilecpp
