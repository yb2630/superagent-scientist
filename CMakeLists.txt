cmake_minimum_required(VERSION 3.25)

project(tilecpp VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(TILECPP_BUILD_TESTS "Build the unit test suite" ON)
option(TILECPP_WARNINGS "Enable the project's warning flags" ON)
option(TILECPP_WERROR "Treat compiler warnings as errors" OFF)
option(TILECPP_SANITIZE
       "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(TILECPP_SANITIZE_THREAD
       "Build with ThreadSanitizer (for later concurrency milestones)" OFF)

if(TILECPP_SANITIZE AND TILECPP_SANITIZE_THREAD)
  message(FATAL_ERROR
          "TILECPP_SANITIZE (ASan+UBSan) and TILECPP_SANITIZE_THREAD (TSan) "
          "are mutually exclusive; use separate build directories.")
endif()

# ---------------------------------------------------------------------------
# Toolchain / language settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

if(TILECPP_WARNINGS)
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
  )
  if(TILECPP_WERROR)
    add_compile_options(-Werror)
  endif()
endif()

if(TILECPP_SANITIZE)
  add_compile_options(
    -fsanitize=address,undefined
    -fno-sanitize-recover=undefined
    -fno-omit-frame-pointer
    -g
  )
  add_link_options(-fsanitize=address,undefined)
endif()

if(TILECPP_SANITIZE_THREAD)
  add_compile_options(
    -fsanitize=thread
    -fno-omit-frame-pointer
    -g
  )
  add_link_options(-fsanitize=thread)
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(tilecpp STATIC
  src/error.cpp
  src/e4m3.cpp
)
add_library(tilecpp::tilecpp ALIAS tilecpp)

target_include_directories(tilecpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(tilecpp PUBLIC cxx_std_23)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(TILECPP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
