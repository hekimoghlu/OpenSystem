/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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

//===--- GenericMetadataBuilder.h - Math utilities. -------------*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
//
// Utility functions for math operations.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BASIC_MATH_UTILS_H
#define LANGUAGE_BASIC_MATH_UTILS_H

#include "Compiler.h"
#include <cstddef>
#include <cstdint>

#if LANGUAGE_COMPILER_IS_MSVC
#include <intrin.h>
#endif

namespace language {

/// Round the given value up to the given alignment, as a power of two.
template <class T>
static inline constexpr T roundUpToAlignment(T offset, T alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}

/// Round the given value up to the given alignment, expressed as a mask (a
/// power of two minus one).
static inline size_t roundUpToAlignMask(size_t size, size_t alignMask) {
  return (size + alignMask) & ~alignMask;
}

static inline unsigned popcount(unsigned value) {
#if LANGUAGE_COMPILER_IS_MSVC && (defined(_M_IX86) || defined(_M_X64))
  // The __popcnt intrinsic is only available when targetting x86{_64} with MSVC.
  return __popcnt(value);
#elif __has_builtin(__builtin_popcount)
  return __builtin_popcount(value);
#else
  // From toolchain/ADT/bit.h which the runtime doesn't have access to (yet?)
  uint32_t v = value;
  v = v - ((v >> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
  return int(((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24);
#endif
}

} // namespace language

#endif // #ifndef LANGUAGE_BASIC_MATH_UTILS_H
