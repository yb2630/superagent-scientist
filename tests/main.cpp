#include <cstdio>
#include <exception>

#include "test_harness.hpp"

int main() {
  const auto& tests = tilecpp_test::registry();
  std::printf("Running %zu test cases\n", tests.size());
  std::fflush(stdout);

  int failures = 0;
  for (const auto& test : tests) {
    std::printf("[ RUN      ] %s\n", test.name);
    std::fflush(stdout);
    try {
      test.function();
      std::printf("[       OK ] %s\n", test.name);
    } catch (const std::exception& exception) {
      ++failures;
      std::printf("[  FAILED  ] %s\n             %s\n", test.name,
                  exception.what());
    }
    std::fflush(stdout);
  }

  std::printf("%zu test cases, %d failure(s)\n", tests.size(), failures);
  return failures == 0 ? 0 : 1;
}
