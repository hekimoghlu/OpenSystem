/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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
#ifndef __X86GPRINTRIN_H
#error "Never use <usermsrintrin.h> directly; include <x86gprintrin.h> instead."
#endif // __X86GPRINTRIN_H

#ifndef __USERMSRINTRIN_H
#define __USERMSRINTRIN_H
#ifdef __x86_64__

/// Reads the contents of a 64-bit MSR specified in \a __A into \a dst.
///
/// This intrinsic corresponds to the <c> URDMSR </c> instruction.
/// \param __A
///    An unsigned long long.
///
/// \code{.operation}
///    DEST := MSR[__A]
/// \endcode
static __inline__ unsigned long long
    __attribute__((__always_inline__, __nodebug__, __target__("usermsr")))
    _urdmsr(unsigned long long __A) {
  return __builtin_ia32_urdmsr(__A);
}

/// Writes the contents of \a __B into the 64-bit MSR specified in \a __A.
///
/// This intrinsic corresponds to the <c> UWRMSR </c> instruction.
///
/// \param __A
///    An unsigned long long.
/// \param __B
///    An unsigned long long.
///
/// \code{.operation}
///    MSR[__A] := __B
/// \endcode
static __inline__ void
    __attribute__((__always_inline__, __nodebug__, __target__("usermsr")))
    _uwrmsr(unsigned long long __A, unsigned long long __B) {
  return __builtin_ia32_uwrmsr(__A, __B);
}

#endif // __x86_64__
#endif // __USERMSRINTRIN_H
