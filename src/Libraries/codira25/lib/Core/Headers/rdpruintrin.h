/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
#if !defined __X86INTRIN_H
#error "Never use <rdpruintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __RDPRUINTRIN_H
#define __RDPRUINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS \
  __attribute__((__always_inline__, __nodebug__,  __target__("rdpru")))


/// Reads the content of a processor register.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> RDPRU </c> instruction.
///
/// \param reg_id
///    A processor register identifier.
static __inline__ unsigned long long __DEFAULT_FN_ATTRS
__rdpru (int reg_id)
{
  return __builtin_ia32_rdpru(reg_id);
}

#define __RDPRU_MPERF 0
#define __RDPRU_APERF 1

/// Reads the content of processor register MPERF.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic generates instruction <c> RDPRU </c> to read the value of
/// register MPERF.
#define __mperf() __builtin_ia32_rdpru(__RDPRU_MPERF)

/// Reads the content of processor register APERF.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic generates instruction <c> RDPRU </c> to read the value of
/// register APERF.
#define __aperf() __builtin_ia32_rdpru(__RDPRU_APERF)

#undef __DEFAULT_FN_ATTRS

#endif /* __RDPRUINTRIN_H */
