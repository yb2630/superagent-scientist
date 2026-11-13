#pragma once

/// \file
/// Owning storage.
///
/// `owning_buffer<T>` is the only type in TileCpp that owns memory.  It is
/// intentionally a thin RAII wrapper around a contiguous, value-initialized
/// array; all numerical operations work on `matrix_view<T>` instead.  A buffer
/// can be larger than a view, which is how padded row strides are supported.

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

namespace tilecpp {

/// Contiguous owning array of `count` value-initialized elements.
template <class T>
class owning_buffer {
 public:
  /// Creates an empty buffer.
  owning_buffer() noexcept = default;

  /// Allocates `count` value-initialized elements.
  explicit owning_buffer(std::size_t count)
      : data_(std::make_unique<T[]>(count)), size_(count) {}

  owning_buffer(const owning_buffer&) = delete;
  owning_buffer& operator=(const owning_buffer&) = delete;

  owning_buffer(owning_buffer&& other) noexcept
      : data_(std::move(other.data_)), size_(other.size_) {
    other.size_ = 0;
  }

  owning_buffer& operator=(owning_buffer&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      size_ = other.size_;
      other.size_ = 0;
    }
    return *this;
  }

  ~owning_buffer() = default;

  [[nodiscard]] T* data() noexcept { return data_.get(); }
  [[nodiscard]] const T* data() const noexcept { return data_.get(); }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  [[nodiscard]] std::span<T> span() noexcept {
    return {data_.get(), size_};
  }
  [[nodiscard]] std::span<const T> span() const noexcept {
    return {data_.get(), size_};
  }

  [[nodiscard]] T& operator[](std::size_t index) noexcept {
    return data_[index];
  }
  [[nodiscard]] const T& operator[](std::size_t index) const noexcept {
    return data_[index];
  }

  [[nodiscard]] T* begin() noexcept { return data_.get(); }
  [[nodiscard]] T* end() noexcept { return data_.get() + size_; }
  [[nodiscard]] const T* begin() const noexcept { return data_.get(); }
  [[nodiscard]] const T* end() const noexcept { return data_.get() + size_; }

 private:
  std::unique_ptr<T[]> data_{};
  std::size_t size_ = 0;
};

}  // namespace tilecpp
