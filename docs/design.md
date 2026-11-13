# TileCpp design

This document records the architecture and the explicit numerical policies of
milestone 1. It is the reference for the later per-block, SIMD, coroutine and
benchmark milestones.

## 1. Scope

Milestone 1 delivers a dependency-free, scalar numerical core:

- owning storage and validated, non-owning row-major matrix views (possibly
  padded strides);
- OCP/ONNX FP8 E4M3FN encode/decode with explicit rounding, saturation and
  special-value behavior;
- per-row scaling, expressed through a scaling-policy interface that a
  per-block policy can implement later;
- row-wise quantize/dequantize kernels parameterized by numerical, scaling and
  execution policies;
- runtime validation through `std::expected`, compile-time validation through
  concepts and `static_assert`;
- an independent Python reference model with committed deterministic vectors;
- AddressSanitizer + UndefinedBehaviorSanitizer (and a ThreadSanitizer
  configuration for later concurrency work).

Out of scope for this milestone: SIMD, per-block scaling, `stdexec`/coroutines,
thread pools, benchmark measurements and fused SwiGLU. The interfaces are
shaped so those additions do not require rewriting the numerical core.

## 2. Layers

```text
 owning_buffer<T>          owns contiguous memory, no arithmetic
        |
 matrix_view<T>            non-owning row-major view (rows, columns, stride)
        |
 quantize_rows /           row/block loops; owns validation and error mapping
 dequantize_rows
        |            \
 numerical policy   scaling policy        execution policy
 (e4m3fn)           (per_row_scale)       (sequential, later thread pool)
```

- **Numerical computation** lives behind a numerical-policy type: storage type,
  metadata and `encode(float) -> storage` / `decode(storage) -> float`.
- **Memory layout** lives in `matrix_view`: a pointer, a logical shape and a row
  stride. Views know nothing about numerical formats or execution.
- **Execution** lives in an execution-policy type exposing
  `for_each_row(rows, fn)`. The kernel contains no threading primitives.
- **Scaling** lives in a scaling-policy type. The kernel knows only how to tile
  a row into `block_count` segments of at most `block_extent` elements, call
  `compute` per segment and store the scales row-major.

These axes are independent template parameters. A mismatch that can be detected
from types (wrong output element type, policy missing a required operation,
unsupported execution policy) fails to compile through concepts. Runtime
properties (shape, stride, storage, scale values) are checked with
`std::expected`.

## 3. Ownership and views

`owning_buffer<T>` is the only owner. It allocates `count` value-initialized
elements and exposes `data()`, `size()` and spans. It is movable, not copyable.

`matrix_view<T>` is a handle to caller-owned storage. It is constructed only
through the validating factory:

```cpp
auto view = matrix_view<const float>::create(storage, rows, columns, stride);
```

Validation happens once, at view creation:

- `stride >= columns` (otherwise `invalid_stride`);
- the required number of elements is `(rows - 1) * stride + columns` for a
  non-empty shape and `0` for an empty shape, with overflow detection
  (`invalid_dimensions`);
- the supplied storage is large enough (`insufficient_storage`).

A view exposes `row(r)` as a span of exactly `columns` elements. Algorithms
therefore cannot accidentally read or write padding, and the last row does not
need trailing padding storage. Empty views (`rows == 0` or `columns == 0`) are
valid and need no storage.

`T` may be const: `matrix_view<const float>` and
`matrix_view<const std::uint8_t>` are read-only views. Conversion from a
mutable view to a const view is implicit; the reverse is rejected.

This milestone fixes the layout to contiguous, possibly padded, row-major.
Adding a layout policy later (column-major, tiled) means changing the view
parameterization, not the numeric kernels.

## 4. FP8 E4M3FN numerical policy

### 4.1 Encoding

| field    | bits | meaning                          |
|----------|------|----------------------------------|
| sign     | 1    | `0` non-negative, `1` negative   |
| exponent | 4    | bias 7                           |
| mantissa | 3    | implicit leading 1 for normals   |

- Normal encodings (`exp` in `1..15`, except the NaN pattern):
  `value = (1 + mantissa/8) * 2^(exp - 7)`.
- Subnormal encodings (`exp == 0`): `value = (mantissa/8) * 2^-6 =
  mantissa * 2^-9`.
- `0x7F` (`exp = 15`, `mantissa = 7`) is NaN; `0xFF` is the negative NaN
  encoding. There are no infinities.
- `0x7E` is `+448`, the largest finite magnitude. `0x08` is the smallest
  normal value, `2^-6`; `0x01` is the smallest subnormal, `2^-9`.

This is the OCP FP8 / ONNX `E4M3FN` variant. The `FN` suffix (finite, NaN-only)
is important: other E4M3 variants differ in bias, NaN encoding or saturation
range. The choice is recorded here because milestone 1 needed one concrete
variant.

### 4.2 Encoding rules

`e4m3fn::encode(float)` is total and follows this specification:

1. NaN and `±Inf` encode to the canonical NaN `0x7F`. Non-finite inputs have
   no finite representation, and a single canonical NaN keeps bit patterns
   deterministic.
2. `±0` preserve their sign: `0x00` and `0x80`.
3. Finite magnitudes above `448` saturate to `±448` (`0x7E`/`0xFE`). This is
   saturating conversion, not overflow to NaN.
4. Everything else uses **round-to-nearest, ties-to-even** on the exact value
   of the input `float`. The implementation splits the magnitude into its
   binary exponent and a three-bit mantissa. A tie rounds the E4M3 encoding to
   the even one. For example, the midpoint between `1.0` (`0x38`) and `1.125`
   (`0x39`) rounds to `1.0`; the midpoint between `1.125` (`0x39`) and `1.25`
   (`0x3A`) rounds to `1.25`.
5. Values below the normal range are rounded on the `2^-9` subnormal grid.
   Values below half the smallest subnormal round to signed zero.

`decode` is exact by construction: normal values are computed with `ldexp` and
exact three-bit mantissas, and subnormals are `mantissa * 2^-9`. Decoding
`0x7F`/`0xFF` yields a quiet NaN.

Quantization divides by the scale before calling `encode`. The division is
performed in double precision and rounded once to float before E4M3 rounding;
that explicit single-precision rounding point is mirrored by the Python
reference so the committed vectors are bit-reproducible. Dequantization
multiplies the decoded float by the scale, also rounded once to float.

## 5. Scaling policy and per-row scales

The kernel tiles each row into blocks:

```text
scales[row * block_count(columns) + block] = ScalePolicy::compute(row segment)
```

- `block_count(columns)` is the number of scales per row.
- `block_extent(columns)` is the maximum segment length; segments are
  `[0, extent)`, `[extent, 2*extent)`, ... with the last one truncated.
- For non-empty rows the policy must satisfy
  `block_count == ceil(columns / extent)`; the kernel validates this and
  rejects inconsistent policies with `unsupported_input`.
- `compute` must return a finite, strictly positive scale. The kernel
  validates this and returns `invalid_scale` otherwise.

`per_row_scale` implements the milestone-1 policy:

```text
max_abs = max |x| over finite values in the logical row (padding excluded)
scale   = max_abs / 448, or 1 if max_abs == 0
```

Consequences:

- Padding elements never influence a scale.
- NaN/Inf elements are encoded as NaN but do not influence the scale.
- Rows with no finite non-zero value (all zeros or all non-finite) use scale
  `1`, so zeros encode to signed zero.
- If `max_abs / 448` underflows to `0` (only possible for denormal-sized
  rows), the policy uses the smallest positive float instead of dividing by
  zero.
- Dequantization requires finite, strictly positive scales for the same
  reason.

Per-block scaling (milestone 2) implements the same three static functions with
an extent such as 128. The kernel loop and scale layout already support it;
only the policy and its tests are new.

## 6. Runtime error model

Fallible operations return `std::expected<T, tilecpp::error>`. `error` carries
an `error_code` and a message:

| code                   | meaning                                                        |
|------------------------|----------------------------------------------------------------|
| `invalid_dimensions`   | shape mismatch, scale-count mismatch, or size overflow         |
| `invalid_stride`       | stride smaller than the logical column count                   |
| `insufficient_storage` | storage shorter than `(rows-1)*stride + columns`               |
| `invalid_scale`        | a scale is not finite and strictly positive                    |
| `unsupported_input`    | a policy/layout combination this build cannot support          |

Validation order in the kernels: matching logical shapes, scaling-policy
consistency and scale-count overflow, exact scale-buffer size, then the empty
fast path. Scale values are validated while rows execute. A late failure can
leave partially written output; the returned error distinguishes that from a
successful result. Built-in quantization policies cannot produce invalid
scales, but dequantization of caller-supplied scales can fail mid-way. Row-local
error slots make the reporting data-race free for a concurrent execution
policy.

Compile-time contracts:

- `numerical_policy`: storage type, `encode`, `decode`, `max_finite`, `name`;
- `scale_policy`: `compute`, `block_count`, `block_extent`, `name`, plus a
  numerical policy providing `max_finite`;
- `execution_policy`: `for_each_row`;
- `matrix_view` parameter types enforce const-correctness (a mutable view
  converts to a read-only view, not vice versa).

Statically knowable incompatibilities (e.g. passing a `float` output view to an
E4M3 kernel, or an execution policy without `for_each_row`) are rejected by the
compiler rather than deferred to runtime.

## 7. Execution policies

An execution policy implements:

```cpp
template <class Function>
void for_each_row(std::size_t rows, Function&& function) const;
```

The contract is: invoke `function(row)` exactly once for every row index and
return only after all invocations complete. Rows own disjoint output bytes,
scale slots and padding, so executing rows concurrently is safe at the kernel
level. Milestone 1 ships `sequential`; the thread-pool policy in milestone 3
implements the same contract. Error slots are row-local for the same reason.

## 8. Testing and reference vectors

The C++ suite is dependency-free (`tests/test_harness.hpp`). It covers:

- decoding of all 256 FP8 bit patterns and encoding around every rounding
  boundary (all neighbours of every midpoint), underflow, subnormals,
  saturation, signed zero, NaN/Inf and seeded random values;
- empty inputs, padded input/output strides, padding preservation,
  invalid shapes, strides, storage sizes, scale counts and scale values;
- per-row scale behavior, zero rows, denormal scales and invalid custom scale
  policies;
- a compile-time section (`static_assert`) for policy concepts.

The Python reference (`tools/tilecpp_reference.py`, standard library only) is
deliberately independent of the C++ algorithm: encoding decisions are made with
exact rational arithmetic (`fractions.Fraction`) over all finite E4M3FN
magnitudes, with exact tie handling. It mirrors the two documented
single-precision rounding points (scale division and normalization) so the
vectors can be compared bit-exactly. `tools/generate_test_vectors.py` writes:

- `tests/data/fp8_decode.tsv`: all 256 decodings;
- `tests/data/fp8_encode.tsv`: every representable value, every rounding
  midpoint and its float32 neighbours, special values and seeded random values;
- `tests/data/quantize.tsv`: matrices with padded strides, huge padding values,
  zero rows, special values, denormal scales and saturation cases, with
  expected scales, quantized bytes and dequantized bits.

The suite is also built and run with
`-DTILECPP_SANITIZE=ON` (ASan + UBSan, `-fno-sanitize-recover=undefined`) and
`-DTILECPP_SANITIZE_THREAD=ON` (TSan; meaningful once concurrency lands).

## 9. Decisions and limitations

- **No external dependencies.** The project uses the C++23 standard library
  only: `std::expected`, `std::span`, concepts and `std::bit_cast`. Test
  vectors are plain text rather than JSON to keep the test harness
  dependency-free.
- **Custom matrix view instead of `std::mdspan`.** The type carries the exact
  validation and error reporting the project needs, has no layout-policy
  parameter yet, and keeps ownership versus views explicit. It can be replaced
  by an `mdspan` alias once layout policies become a milestone requirement.
- **Alignment.** `owning_buffer` guarantees only natural alignment; a SIMD
  milestone may add an aligned allocator. This is recorded so the SIMD work
  does not silently assume 32/64-byte alignment.
- **Float32 only.** Inputs and dequantized outputs are `float`. FP16/BF16
  inputs are later work.
- **Partial output on late scale failure.** Validation that requires computing
  scales happens row by row; a failing custom policy may leave earlier rows
  written. The returned error is authoritative. Built-in policies cannot fail.
- **Single header-level numerical policy.** Additional FP8 variants (E5M2,
  E4M3FNUZ) are policy additions, not kernel changes.

## 10. Milestone plan

1. **This milestone:** scaffold; owning storage and views; scalar E4M3FN
   quantize/dequantize with per-row scales; validation; Python reference and
   tests; ASan/UBSan and TSan build configurations.
2. Per-block scaling policy and SIMD backend; specialized-vs-generic
   performance comparison.
3. `stdexec` sender/coroutine batch interface, thread-pool execution policy,
   cancellation and ownership tests; sequential/sender/coroutine comparison.
4. Benchmark harness and measurements (throughput, median/p95 latency, peak
   memory, allocations), design-trade-off report, fused SwiGLU stretch goal.

## 11. Reproducing milestone-1 results

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTILECPP_WERROR=ON
cmake --build build
./build/tests/tilecpp_tests

cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DTILECPP_SANITIZE=ON -DTILECPP_WERROR=ON
cmake --build build-asan
ctest --test-dir build-asan --output-on-failure

cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DTILECPP_SANITIZE_THREAD=ON
cmake --build build-tsan
ctest --test-dir build-tsan --output-on-failure
```
