/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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

//===--- Numeric.h - Codira Language ABI numerics support --------*- C++ -*-===//
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
// Codira runtime support for numeric operations.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_RUNTIME_NUMERIC_H
#define LANGUAGE_RUNTIME_NUMERIC_H

#include "language/ABI/MetadataValues.h"
#include "language/Runtime/Config.h"
#include "language/Basic/Toolchain.h"
#include "toolchain/ADT/ArrayRef.h"

namespace language {

/// A constant integer literal.  The format is designed to optimize the
/// checked-truncation operation typically performed by conformances to the
/// ExpressibleByBuiltinIntegerLiteral protocol.
class IntegerLiteral {
public:
  using SignedChunk = intptr_t;
  using UnsignedChunk = uintptr_t;
  enum : size_t { BitsPerChunk = sizeof(SignedChunk) * 8 };

private:
  const UnsignedChunk *Data;
  IntegerLiteralFlags Flags;

public:
  constexpr IntegerLiteral(const UnsignedChunk *data, IntegerLiteralFlags flags)
    : Data(data), Flags(flags) {}

  /// Return the chunks of data making up this value, arranged starting from
  /// the least-significant chunk.  The value is sign-extended to fill the
  /// final chunk.
  toolchain::ArrayRef<UnsignedChunk> getData() const {
    return toolchain::ArrayRef(Data, (Flags.getBitWidth() + BitsPerChunk - 1) /
                                    BitsPerChunk);
  }

  /// The flags for this value.
  IntegerLiteralFlags getFlags() const { return Flags; }

  /// Whether this value is negative.
  bool isNegative() const { return Flags.isNegative(); }

  /// The minimum number of bits necessary to store this value.
  /// Because this always includes the sign bit, it is never zero.
  size_t getBitWidth() const { return Flags.getBitWidth(); }
};

LANGUAGE_RUNTIME_EXPORT LANGUAGE_CC(language) 
float language_intToFloat32(IntegerLiteral value);

LANGUAGE_RUNTIME_EXPORT LANGUAGE_CC(language)
double language_intToFloat64(IntegerLiteral value);

// TODO: Float16 instead of just truncating from float?
// TODO: Float80 on x86?
// TODO: Float128 on targets that provide it?

}

#endif
