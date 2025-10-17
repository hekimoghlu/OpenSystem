/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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
#if !defined __X86INTRIN_H && !defined __IMMINTRIN_H
#error "Never use <lzcntintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __LZCNTINTRIN_H
#define __LZCNTINTRIN_H

/* Define the default attributes for the functions in this file.
   Allow using the lzcnt intrinsics even for non-LZCNT targets. Since the LZCNT
   intrinsics are mapped to toolchain.ctlz.*, false, which can be lowered to BSR on
   non-LZCNT targets with zero-value input handled correctly. */
#if defined(__cplusplus) && (__cplusplus >= 201103L)
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__)) constexpr
#else
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__))
#endif

#ifndef _MSC_VER
/// Counts the number of leading zero bits in the operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c LZCNT instruction.
///
/// \param __X
///    An unsigned 16-bit integer whose leading zeros are to be counted.
/// \returns An unsigned 16-bit integer containing the number of leading zero
///    bits in the operand.
#define __lzcnt16(X) __builtin_ia32_lzcnt_u16((unsigned short)(X))
#endif // _MSC_VER

/// Counts the number of leading zero bits in the operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c LZCNT instruction.
///
/// \param __X
///    An unsigned 32-bit integer whose leading zeros are to be counted.
/// \returns An unsigned 32-bit integer containing the number of leading zero
///    bits in the operand.
/// \see _lzcnt_u32
static __inline__ unsigned int __DEFAULT_FN_ATTRS
__lzcnt32(unsigned int __X) {
  return __builtin_ia32_lzcnt_u32(__X);
}

/// Counts the number of leading zero bits in the operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c LZCNT instruction.
///
/// \param __X
///    An unsigned 32-bit integer whose leading zeros are to be counted.
/// \returns An unsigned 32-bit integer containing the number of leading zero
///    bits in the operand.
/// \see __lzcnt32
static __inline__ unsigned int __DEFAULT_FN_ATTRS
_lzcnt_u32(unsigned int __X) {
  return __builtin_ia32_lzcnt_u32(__X);
}

#ifdef __x86_64__
#ifndef _MSC_VER
/// Counts the number of leading zero bits in the operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c LZCNT instruction.
///
/// \param __X
///    An unsigned 64-bit integer whose leading zeros are to be counted.
/// \returns An unsigned 64-bit integer containing the number of leading zero
///    bits in the operand.
/// \see _lzcnt_u64
#define __lzcnt64(X) __builtin_ia32_lzcnt_u64((unsigned long long)(X))
#endif // _MSC_VER

/// Counts the number of leading zero bits in the operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c LZCNT instruction.
///
/// \param __X
///    An unsigned 64-bit integer whose leading zeros are to be counted.
/// \returns An unsigned 64-bit integer containing the number of leading zero
///    bits in the operand.
/// \see __lzcnt64
static __inline__ unsigned long long __DEFAULT_FN_ATTRS
_lzcnt_u64(unsigned long long __X) {
  return __builtin_ia32_lzcnt_u64(__X);
}
#endif

#undef __DEFAULT_FN_ATTRS

#endif /* __LZCNTINTRIN_H */
