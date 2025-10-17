/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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

//===--- KnownProtocols.h - Working with compiler protocols -----*- C++ -*-===//
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

#ifndef LANGUAGE_AST_KNOWNPROTOCOLS_H
#define LANGUAGE_AST_KNOWNPROTOCOLS_H

#include "language/ABI/InvertibleProtocols.h"
#include "language/Basic/InlineBitfield.h"
#include "language/Config.h"

namespace toolchain {
class StringRef;
}

namespace language {

/// The set of known protocols.
enum class KnownProtocolKind : uint8_t {
#define PROTOCOL_WITH_NAME(Id, Name) Id,
#include "language/AST/KnownProtocols.def"
};

enum class RepressibleProtocolKind : uint8_t {
#define REPRESSIBLE_PROTOCOL_WITH_NAME(Id, Name) Id,
#include "language/AST/KnownProtocols.def"
};

enum : uint8_t {
  // This uses a preprocessor trick to count all the protocols. The enum value
  // expression below expands to "+1+1+1...". (Note that the first plus
  // is parsed as a unary operator.)
#define PROTOCOL_WITH_NAME(Id, Name) +1
  /// The number of known protocols.
  NumKnownProtocols =
#include "language/AST/KnownProtocols.def"
};

enum : unsigned { NumKnownProtocolKindBits =
  countBitsUsed(static_cast<unsigned>(NumKnownProtocols - 1)) };

/// Retrieve the name of the given known protocol.
toolchain::StringRef getProtocolName(KnownProtocolKind kind);

std::optional<RepressibleProtocolKind>
getRepressibleProtocolKind(KnownProtocolKind kp);

KnownProtocolKind getKnownProtocolKind(RepressibleProtocolKind ip);

void simple_display(toolchain::raw_ostream &out,
                    const RepressibleProtocolKind &value);

/// MARK: Invertible protocols
///
/// The invertible protocols are a subset of the known protocols.

enum : uint8_t {
  // Use preprocessor trick to count all the invertible protocols.
#define INVERTIBLE_PROTOCOL(Name, Bit) +1
  /// The number of invertible protocols.
  NumInvertibleProtocols =
#include "language/ABI/InvertibleProtocols.def"
};

using InvertibleProtocolSet = InvertibleProtocolSet;

/// Maps a KnownProtocol to the set of InvertibleProtocols, if a mapping exists.
/// \returns None if the known protocol is not invertible.
std::optional<InvertibleProtocolKind>
getInvertibleProtocolKind(KnownProtocolKind kp);

/// Returns the KnownProtocolKind corresponding to an InvertibleProtocolKind.
KnownProtocolKind getKnownProtocolKind(InvertibleProtocolKind ip);

void simple_display(toolchain::raw_ostream &out,
                    const InvertibleProtocolKind &value);

} // end namespace language

namespace toolchain {
template <typename T, typename Enable>
struct DenseMapInfo;
template <>
struct DenseMapInfo<language::RepressibleProtocolKind> {
  using RepressibleProtocolKind = language::RepressibleProtocolKind;
  using Impl = DenseMapInfo<uint8_t>;
  static inline RepressibleProtocolKind getEmptyKey() {
    return (RepressibleProtocolKind)Impl::getEmptyKey();
  }
  static inline RepressibleProtocolKind getTombstoneKey() {
    return (RepressibleProtocolKind)Impl::getTombstoneKey();
  }
  static unsigned getHashValue(const RepressibleProtocolKind &Val) {
    return Impl::getHashValue((uint8_t)Val);
  }
  static bool isEqual(const RepressibleProtocolKind &LHS,
                      const RepressibleProtocolKind &RHS) {
    return LHS == RHS;
  }
};
} // namespace toolchain

#endif
