/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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

//===--- Numeric.cpp - Codira Language ABI numerics support ----------------===//
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
// Implementations of the numeric-support ABI functions.
//
//===----------------------------------------------------------------------===//

#include "language/Runtime/Numeric.h"

using namespace language;

/// Convert an integer literal to the floating-point type T.
template <class T>
static T convert(IntegerLiteral value) {
  using SignedChunk = IntegerLiteral::SignedChunk;
  using UnsignedChunk = IntegerLiteral::UnsignedChunk;

  auto data = value.getData();
  assert(!data.empty() && "always require at least one chunk");

  // The single-word case is the easiest.
  if (data.size() == 1) {
    return T(SignedChunk(data[0]));
  }

  // In two's complement, only the topmost chunk is really signed;
  // everything else is added to that as an unsigned value.
  static_assert(IntegerLiteral::BitsPerChunk == 32 ||
                IntegerLiteral::BitsPerChunk == 64,
                "expected either 32-bit or 64-bit chunking");
  T chunkFactor = (IntegerLiteral::BitsPerChunk == 32 ? 0x1p32 : 0x1p64);

  T result = UnsignedChunk(data[0]);
  T scale = chunkFactor;
  for (size_t i = 1, e = data.size() - 1; i != e; ++i) {
    result += UnsignedChunk(data[i]) * scale;
    scale *= chunkFactor;
  }
  result += SignedChunk(data.back()) * scale;

  return result;
}

float language::language_intToFloat32(IntegerLiteral value) {
  return convert<float>(value);
}

double language::language_intToFloat64(IntegerLiteral value) {
  return convert<double>(value);
}
