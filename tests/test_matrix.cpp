#include <cstddef>
#include <limits>
#include <span>
#include <utility>

#include "test_harness.hpp"
#include "tilecpp/buffer.hpp"
#include "tilecpp/matrix_view.hpp"

namespace {

using tilecpp::error_code;
using tilecpp::matrix_view;
using tilecpp::owning_buffer;

TILECPP_TEST(buffer_basics) {
  owning_buffer<float> buffer(4);
  CHECK(buffer.size() == 4U);
  CHECK(!buffer.empty());
  CHECK(buffer[0] == 0.0F);  // value-initialized
  CHECK(buffer[3] == 0.0F);

  buffer[1] = 2.5F;
  auto span = buffer.span();
  CHECK(span.size() == 4U);
  CHECK(span[1] == 2.5F);

  const owning_buffer<float>& const_buffer = buffer;
  CHECK(const_buffer[1] == 2.5F);
  CHECK(const_buffer.span()[1] == 2.5F);

  owning_buffer<float> empty;
  CHECK(empty.empty());
  CHECK(empty.size() == 0U);
  CHECK(empty.data() == nullptr);
  CHECK(empty.span().empty());

  owning_buffer<float> zero_sized(0);
  CHECK(zero_sized.empty());
}

TILECPP_TEST(buffer_move_semantics) {
  owning_buffer<float> source(3);
  source[0] = 7.0F;

  owning_buffer<float> moved(std::move(source));
  CHECK(moved.size() == 3U);
  CHECK(moved[0] == 7.0F);
  CHECK(source.empty());
  CHECK(source.data() == nullptr);
  CHECK(source.span().empty());

  owning_buffer<float> assigned;
  assigned = std::move(moved);
  CHECK(assigned.size() == 3U);
  CHECK(assigned[0] == 7.0F);
  CHECK(moved.empty());
  CHECK(moved.data() == nullptr);
}

TILECPP_TEST(matrix_view_contiguous_and_padded) {
  owning_buffer<float> storage(10);
  const auto contiguous =
      CHECKED(matrix_view<float>::create(storage.span(), 2, 3, 3)).value();
  CHECK(contiguous.rows() == 2U);
  CHECK(contiguous.columns() == 3U);
  CHECK(contiguous.stride() == 3U);
  CHECK(contiguous.element_count() == 6U);
  CHECK(!contiguous.empty());
  CHECK(contiguous.row(0).size() == 3U);
  CHECK(contiguous.row(1).data() == storage.data() + 3);

  contiguous(0, 0) = 1.5F;
  contiguous(1, 2) = 6.5F;
  CHECK(storage[0] == 1.5F);
  CHECK(storage[5] == 6.5F);

  const auto padded =
      CHECKED(matrix_view<float>::create(storage.span(), 2, 3, 5)).value();
  CHECK(padded.row(0).size() == 3U);
  CHECK(padded.row(1).data() == storage.data() + 5);
  CHECK(padded(1, 0) == storage[5]);

  const auto read_only = CHECKED(matrix_view<const float>::create(
                                     std::span<const float>(storage.span()), 2,
                                     3, 3))
                             .value();
  CHECK(read_only(0, 0) == 1.5F);
  CHECK(read_only.row(0).size() == 3U);
}

TILECPP_TEST(error_model_formatting) {
  CHECK(tilecpp::to_string(error_code::invalid_stride) == "invalid_stride");
  CHECK(tilecpp::to_string(error_code::insufficient_storage) ==
        "insufficient_storage");
  const tilecpp::error example{error_code::invalid_scale,
                               "scale must be finite and positive"};
  CHECK(tilecpp::format(example) ==
        "invalid_scale: scale must be finite and positive");
}

TILECPP_TEST(matrix_view_empty_shapes) {
  const std::span<float> no_storage;
  const auto empty =
      CHECKED(matrix_view<float>::create(no_storage, 0, 0, 0)).value();
  CHECK(empty.empty());
  CHECK(empty.rows() == 0U);
  CHECK(empty.columns() == 0U);
  CHECK(empty.element_count() == 0U);

  const auto zero_columns =
      CHECKED(matrix_view<float>::create(no_storage, 3, 0, 0)).value();
  CHECK(zero_columns.empty());
  CHECK(zero_columns.rows() == 3U);
  CHECK(zero_columns.columns() == 0U);
  CHECK(zero_columns.row(0).empty());

  const auto zero_rows =
      CHECKED(matrix_view<float>::create(no_storage, 0, 4, 4)).value();
  CHECK(zero_rows.empty());
  CHECK(zero_rows.rows() == 0U);
  CHECK(zero_rows.columns() == 4U);
}

TILECPP_TEST(matrix_view_rejects_invalid_layouts) {
  owning_buffer<float> storage(10);

  CHECK_ERROR(
      (matrix_view<float>::create(storage.span(), 2, 3, 2)),
      error_code::invalid_stride);

  owning_buffer<float> too_small(5);
  CHECK_ERROR(
      (matrix_view<float>::create(too_small.span(), 2, 3, 3)),
      error_code::insufficient_storage);

  owning_buffer<float> one_short(7);
  CHECK_ERROR(
      (matrix_view<float>::create(one_short.span(), 2, 3, 5)),
      error_code::insufficient_storage);

  constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
  CHECK_ERROR(
      (matrix_view<float>::create(storage.span(), max, 2, 2)),
      error_code::invalid_dimensions);
}

}  // namespace
