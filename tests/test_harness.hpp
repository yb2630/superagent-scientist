#pragma once

/// Minimal dependency-free test harness.
///
/// Tests are registered at static-initialization time; main() runs them in
/// registration order.  A failed CHECK throws a `failure` carrying file/line
/// and an optional detail string, which main() reports before moving on.

#include <cstddef>
#include <exception>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tilecpp/error.hpp"

namespace tilecpp_test {

struct test_case {
  const char* name;
  void (*function)();
};

inline std::vector<test_case>& registry() {
  static std::vector<test_case> tests;
  return tests;
}

struct registrar {
  registrar(const char* name, void (*function)()) {
    registry().push_back({name, function});
  }
};

class failure : public std::exception {
 public:
  explicit failure(std::string message) : message_(std::move(message)) {}

  [[nodiscard]] const char* what() const noexcept override {
    return message_.c_str();
  }

 private:
  std::string message_;
};

inline void check(bool condition, const char* expression,
                  const std::string& detail, const char* file, int line) {
  if (condition) {
    return;
  }
  std::ostringstream stream;
  stream << file << ':' << line << ": CHECK(" << expression << ") failed";
  if (!detail.empty()) {
    stream << " -- " << detail;
  }
  throw failure(stream.str());
}

template <class... Args>
std::string make_detail(const Args&... args) {
  std::ostringstream stream;
  (stream << ... << args);
  return stream.str();
}

template <class T>
std::expected<T, tilecpp::error> checked(
    std::expected<T, tilecpp::error> result, const char* expression,
    const char* file, int line) {
  if (!result) {
    throw failure(std::string(file) + ':' + std::to_string(line) + ": " +
                  expression + " failed: " + tilecpp::format(result.error()));
  }
  return result;
}

template <class T>
void check_error(const std::expected<T, tilecpp::error>& result,
                 tilecpp::error_code expected, const char* expression,
                 const char* file, int line) {
  if (result) {
    throw failure(std::string(file) + ':' + std::to_string(line) + ": " +
                  expression + " unexpectedly succeeded");
  }
  if (result.error().code != expected) {
    throw failure(std::string(file) + ':' + std::to_string(line) + ": " +
                  expression + " produced " +
                  std::string(tilecpp::to_string(result.error().code)) +
                  ", expected " + std::string(tilecpp::to_string(expected)) +
                  " (" + result.error().message + ")");
  }
}

}  // namespace tilecpp_test

#define TILECPP_TEST(name)                                            \
  static void name();                                                 \
  static const ::tilecpp_test::registrar name##_registrar(#name,      \
                                                          &name);     \
  static void name()

#define CHECK(condition)                                                  \
  ::tilecpp_test::check(static_cast<bool>(condition), #condition, "",     \
                        __FILE__, __LINE__)

#define CHECK_MSG(condition, ...)                                        \
  ::tilecpp_test::check(static_cast<bool>(condition), #condition,        \
                        ::tilecpp_test::make_detail(__VA_ARGS__),        \
                        __FILE__, __LINE__)

#define CHECKED(expression)                                            \
  ::tilecpp_test::checked((expression), #expression, __FILE__, __LINE__)

#define CHECK_ERROR(expression, expected_code)                            \
  ::tilecpp_test::check_error((expression), (expected_code), #expression, \
                              __FILE__, __LINE__)
