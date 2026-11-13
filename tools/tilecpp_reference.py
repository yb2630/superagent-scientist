"""Independent Python reference model for TileCpp milestone 1.

Standard library only.  The E4M3FN encoding decision is made with exact
rational arithmetic (`fractions.Fraction`): every finite float32 input is
converted to an exact Fraction, and the nearest representable E4M3FN magnitude
is selected, with exact ties resolved towards the even E4M3 encoding.  This is
deliberately a different implementation strategy from the C++ code (which uses
frexp / ldexp / round-to-nearest-even on float significands), so the two agree
only if both implement the documented specification.

The reference also mirrors the two explicit single-precision rounding points of
the C++ kernel:
  - `scale = float32(max_abs / 448)` (double division, then one float cast);
  - `normalized = float32(value / scale)` (double division, then float cast);
  - `dequantized = float32(decode(byte) * scale)`.

Run `tools/generate_test_vectors.py` to regenerate the committed vectors.
"""

from __future__ import annotations

import bisect
import math
import struct
from fractions import Fraction

E4M3_MAX_FINITE = 448.0
E4M3_MAX_FINITE_CODE = 0x7E
E4M3_NAN_CODE = 0x7F
E4M3_SIGN_BIT = 0x80
E4M3_DENORM_MIN = struct.unpack("<f", struct.pack("<I", 1))[0]  # 2^-149


def f32(value: float) -> float:
    """Round a Python float to the nearest binary32 value."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _finite_magnitudes():
    """Exact magnitude of every finite non-negative E4M3FN code (0x00..0x7E)."""
    magnitudes = []
    for code in range(0x7F):
        exponent = (code >> 3) & 0xF
        mantissa = code & 0x7
        if exponent == 0:
            magnitudes.append(Fraction(mantissa, 512))
        elif exponent >= 7:
            numerator = (8 + mantissa) * (1 << (exponent - 7))
            magnitudes.append(Fraction(numerator, 8))
        else:
            denominator = 8 * (1 << (7 - exponent))
            magnitudes.append(Fraction(8 + mantissa, denominator))
    return magnitudes


FINITE_MAGNITUDES = _finite_magnitudes()


def decode(code: int) -> float:
    """Decode one E4M3FN byte to the exactly representable float32 value."""
    code &= 0xFF
    if (code & 0x7F) == 0x7F:
        return math.nan
    magnitude = float(FINITE_MAGNITUDES[code & 0x7F])
    return -magnitude if (code & E4M3_SIGN_BIT) else magnitude


def encode(value: float) -> int:
    """Encode a float32 value: nearest, ties-to-even, saturating, canonical NaN."""
    if math.isnan(value) or math.isinf(value):
        return E4M3_NAN_CODE

    negative = math.copysign(1.0, value) < 0.0
    sign = E4M3_SIGN_BIT if negative else 0
    magnitude = abs(value)
    if magnitude == 0.0:
        return sign
    if magnitude > E4M3_MAX_FINITE:
        return sign | E4M3_MAX_FINITE_CODE

    exact = Fraction(magnitude)
    index = bisect.bisect_left(FINITE_MAGNITUDES, exact)
    if index < len(FINITE_MAGNITUDES) and FINITE_MAGNITUDES[index] == exact:
        code = index
    else:
        lower = index - 1
        upper = index
        lower_distance = exact - FINITE_MAGNITUDES[lower]
        upper_distance = FINITE_MAGNITUDES[upper] - exact
        if lower_distance < upper_distance:
            code = lower
        elif upper_distance < lower_distance:
            code = upper
        else:
            # Round-to-nearest-even: pick the candidate whose encoding is even.
            code = lower if lower % 2 == 0 else upper
    return sign | code


def per_row_scale(row) -> float:
    """Reference per-row scale: max finite magnitude / 448 (or 1 for zero rows)."""
    max_abs = 0.0
    for value in row:
        magnitude = abs(value)
        if math.isfinite(magnitude) and magnitude > max_abs:
            max_abs = magnitude
    if max_abs == 0.0:
        return 1.0
    scale = f32(max_abs / E4M3_MAX_FINITE)
    if scale == 0.0:
        scale = E4M3_DENORM_MIN
    return scale


def quantize_rows(values, rows: int, columns: int, input_stride: int):
    """Quantize a row-major buffer (padding included) with per-row scales.

    Returns (scales, quantized_codes, dequantized_values), each row-major and
    restricted to the logical columns.
    """
    assert len(values) == rows * input_stride
    scales = []
    quantized = []
    dequantized = []
    for row in range(rows):
        start = row * input_stride
        row_values = values[start : start + columns]
        scale = per_row_scale(row_values)
        scales.append(scale)
        row_codes = []
        row_dequantized = []
        for value in row_values:
            if math.isnan(value) or math.isinf(value):
                code = E4M3_NAN_CODE
            else:
                code = encode(f32(value / scale))
            row_codes.append(code)
            decoded = decode(code)
            row_dequantized.append(
                math.nan if math.isnan(decoded) else f32(decoded * scale)
            )
        quantized.append(row_codes)
        dequantized.append(row_dequantized)
    return scales, quantized, dequantized
