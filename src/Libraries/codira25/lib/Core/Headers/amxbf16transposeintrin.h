/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef __IMMINTRIN_H
#error                                                                         \
    "Never use <amxbf16transposeintrin.h> directly; use <immintrin.h> instead."
#endif /* __IMMINTRIN_H */

#ifndef __AMX_BF16TRANSPOSEINTRIN_H
#define __AMX_BF16TRANSPOSEINTRIN_H
#ifdef __x86_64__

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("amx-bf16,amx-transpose")))

/// Compute transpose and dot-product of BF16 (16-bit) floating-point pairs in
///    tiles \a a and \a b, accumulating the intermediate single-precision
///    (32-bit) floating-point elements with elements in \a dst, and store the
///    32-bit result back to tile \a dst.
///
/// \headerfile <immintrin.h>
///
/// \code
/// void _tile_tdpbf16ps (__tile dst, __tile a, __tile b)
/// \endcode
///
/// \code{.operation}
/// FOR m := 0 TO dst.rows - 1
///	tmp := dst.row[m]
///	FOR k := 0 TO (a.colsb / 4) - 1
///		FOR n := 0 TO (dst.colsb / 4) - 1
///			tmp.bf32[n] += FP32(a.row[m].bf16[2*k+0]) *
///					FP32(b.row[k].bf16[2*n+0])
///			tmp.bf32[n] += FP32(a.row[m].bf16[2*k+1]) *
///					FP32(b.row[k].bf16[2*n+1])
///		ENDFOR
///	ENDFOR
///	write_row_and_zero(dst, m, tmp, dst.colsb)
/// ENDFOR
/// zero_upper_rows(dst, dst.rows)
/// zero_tileconfig_start()
/// \endcode
///
/// This intrinsic corresponds to the \c TTDPBF16PS instruction.
///
/// \param dst
///    The destination tile. Max size is 1024 Bytes.
/// \param a
///    The 1st source tile. Max size is 1024 Bytes.
/// \param b
///    The 2nd source tile. Max size is 1024 Bytes.
#define _tile_tdpbf16ps(dst, a, b) __builtin_ia32_ttdpbf16ps((dst), (a), (b))

/// This is internal intrinsic. C/C++ user should avoid calling it directly.
static __inline__ _tile1024i __DEFAULT_FN_ATTRS
_tile_tdpbf16ps_internal(unsigned short m, unsigned short n, unsigned short k,
                         _tile1024i dst, _tile1024i src1, _tile1024i src2) {
  return __builtin_ia32_ttdpbf16ps_internal(m, n, k, dst, src1, src2);
}

/// Compute transpose and dot-product of BF16 (16-bit) floating-point pairs in
///    tiles src0 and src1, accumulating the intermediate single-precision
///    (32-bit) floating-point elements with elements in "dst", and store the
///    32-bit result back to tile "dst".
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> TTDPBF16PS </c> instruction.
///
/// \param dst
///    The destination tile. Max size is 1024 Bytes.
/// \param src0
///    The 1st source tile. Max size is 1024 Bytes.
/// \param src1
///    The 2nd source tile. Max size is 1024 Bytes.
__DEFAULT_FN_ATTRS
static __inline__ void __tile_tdpbf16ps(__tile1024i *dst, __tile1024i src0,
                                        __tile1024i src1) {
  dst->tile = _tile_tdpbf16ps_internal(src0.row, src1.col, src0.col, dst->tile,
                                       src0.tile, src1.tile);
}

#undef __DEFAULT_FN_ATTRS

#endif /* __x86_64__ */
#endif /* __AMX_BF16TRANSPOSEINTRIN_H */
