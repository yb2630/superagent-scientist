add_executable(tilecpp_tests
  main.cpp
  test_fp8.cpp
  test_matrix.cpp
  test_quantize.cpp
  test_reference.cpp
)
target_link_libraries(tilecpp_tests PRIVATE tilecpp)
target_compile_definitions(tilecpp_tests PRIVATE
  TILECPP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

add_test(NAME tilecpp_tests COMMAND tilecpp_tests)
