/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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
#error "Never use <bmi2intrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __BMI2INTRIN_H
#define __BMI2INTRIN_H

/* Define the default attributes for the functions in this file. */
#if defined(__cplusplus) && (__cplusplus >= 201103L)
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __target__("bmi2"))) constexpr
#else
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __target__("bmi2")))
#endif

/// Copies the unsigned 32-bit integer \a __X and zeroes the upper bits
///    starting at bit number \a __Y.
///
/// \code{.operation}
/// i := __Y[7:0]
/// result := __X
/// IF i < 32
///   result[31:i] := 0
/// FI
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c BZHI instruction.
///
/// \param __X
///    The 32-bit source value to copy.
/// \param __Y
///    The lower 8 bits specify the bit number of the lowest bit to zero.
/// \returns The partially zeroed 32-bit value.
static __inline__ unsigned int __DEFAULT_FN_ATTRS
_bzhi_u32(unsigned int __X, unsigned int __Y) {
  return __builtin_ia32_bzhi_si(__X, __Y);
}

/// Deposit (scatter) low-order bits from the unsigned 32-bit integer \a __X
///    into the 32-bit result, according to the mask in the unsigned 32-bit
///    integer \a __Y. All other bits of the result are zero.
///
/// \code{.operation}
/// i := 0
/// result := 0
/// FOR m := 0 TO 31
///   IF __Y[m] == 1
///     result[m] := __X[i]
///     i := i + 1
///   ENDIF
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c PDEP instruction.
///
/// \param __X
///    The 32-bit source value to copy.
/// \param __Y
///    The 32-bit mask specifying where to deposit source bits.
/// \returns The 32-bit result.
static __inline__ unsigned int __DEFAULT_FN_ATTRS
_pdep_u32(unsigned int __X, unsigned int __Y) {
  return __builtin_ia32_pdep_si(__X, __Y);
}

/// Extract (gather) bits from the unsigned 32-bit integer \a __X into the
///    low-order bits of the 32-bit result, according to the mask in the
///    unsigned 32-bit integer \a __Y. All other bits of the result are zero.
///
/// \code{.operation}
/// i := 0
/// result := 0
/// FOR m := 0 TO 31
///   IF __Y[m] == 1
///     result[i] := __X[m]
///     i := i + 1
///   ENDIF
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c PEXT instruction.
///
/// \param __X
///    The 32-bit source value to copy.
/// \param __Y
///    The 32-bit mask specifying which source bits to extract.
/// \returns The 32-bit result.
static __inline__ unsigned int __DEFAULT_FN_ATTRS
_pext_u32(unsigned int __X, unsigned int __Y) {
  return __builtin_ia32_pext_si(__X, __Y);
}

/// Multiplies the unsigned 32-bit integers \a __X and \a __Y to form a
///    64-bit product. Stores the upper 32 bits of the product in the
///    memory at \a __P and returns the lower 32 bits.
///
/// \code{.operation}
/// Store32(__P, (__X * __Y)[63:32])
/// result := (__X * __Y)[31:0]
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c MULX instruction.
///
/// \param __X
///    An unsigned 32-bit multiplicand.
/// \param __Y
///    An unsigned 32-bit multiplicand.
/// \param __P
///    A pointer to memory for storing the upper half of the product.
/// \returns The lower half of the product.
static __inline__ unsigned int __DEFAULT_FN_ATTRS
_mulx_u32(unsigned int __X, unsigned int __Y, unsigned int *__P) {
  unsigned long long __res = (unsigned long long) __X * __Y;
  *__P = (unsigned int)(__res >> 32);
  return (unsigned int)__res;
}

#ifdef  __x86_64__

/// Copies the unsigned 64-bit integer \a __X and zeroes the upper bits
///    starting at bit number \a __Y.
///
/// \code{.operation}
/// i := __Y[7:0]
/// result := __X
/// IF i < 64
///   result[63:i] := 0
/// FI
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c BZHI instruction.
///
/// \param __X
///    The 64-bit source value to copy.
/// \param __Y
///    The lower 8 bits specify the bit number of the lowest bit to zero.
/// \returns The partially zeroed 64-bit value.
static __inline__ unsigned long long __DEFAULT_FN_ATTRS
_bzhi_u64(unsigned long long __X, unsigned long long __Y) {
  return __builtin_ia32_bzhi_di(__X, __Y);
}

/// Deposit (scatter) low-order bits from the unsigned 64-bit integer \a __X
///    into the 64-bit result, according to the mask in the unsigned 64-bit
///    integer \a __Y. All other bits of the result are zero.
///
/// \code{.operation}
/// i := 0
/// result := 0
/// FOR m := 0 TO 63
///   IF __Y[m] == 1
///     result[m] := __X[i]
///     i := i + 1
///   ENDIF
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c PDEP instruction.
///
/// \param __X
///    The 64-bit source value to copy.
/// \param __Y
///    The 64-bit mask specifying where to deposit source bits.
/// \returns The 64-bit result.
static __inline__ unsigned long long __DEFAULT_FN_ATTRS
_pdep_u64(unsigned long long __X, unsigned long long __Y) {
  return __builtin_ia32_pdep_di(__X, __Y);
}

/// Extract (gather) bits from the unsigned 64-bit integer \a __X into the
///    low-order bits of the 64-bit result, according to the mask in the
///    unsigned 64-bit integer \a __Y. All other bits of the result are zero.
///
/// \code{.operation}
/// i := 0
/// result := 0
/// FOR m := 0 TO 63
///   IF __Y[m] == 1
///     result[i] := __X[m]
///     i := i + 1
///   ENDIF
/// ENDFOR
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c PEXT instruction.
///
/// \param __X
///    The 64-bit source value to copy.
/// \param __Y
///    The 64-bit mask specifying which source bits to extract.
/// \returns The 64-bit result.
static __inline__ unsigned long long __DEFAULT_FN_ATTRS
_pext_u64(unsigned long long __X, unsigned long long __Y) {
  return __builtin_ia32_pext_di(__X, __Y);
}

/// Multiplies the unsigned 64-bit integers \a __X and \a __Y to form a
///    128-bit product. Stores the upper 64 bits of the product to the
///    memory addressed by \a __P and returns the lower 64 bits.
///
/// \code{.operation}
/// Store64(__P, (__X * __Y)[127:64])
/// result := (__X * __Y)[63:0]
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c MULX instruction.
///
/// \param __X
///    An unsigned 64-bit multiplicand.
/// \param __Y
///    An unsigned 64-bit multiplicand.
/// \param __P
///    A pointer to memory for storing the upper half of the product.
/// \returns The lower half of the product.
static __inline__ unsigned long long __DEFAULT_FN_ATTRS
_mulx_u64 (unsigned long long __X, unsigned long long __Y,
           unsigned long long *__P) {
  unsigned __int128 __res = (unsigned __int128) __X * __Y;
  *__P = (unsigned long long) (__res >> 64);
  return (unsigned long long) __res;
}

#endif /* __x86_64__  */

#undef __DEFAULT_FN_ATTRS

#endif /* __BMI2INTRIN_H */
