/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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

//===--- Unicode.h - Unicode utilities --------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_BASIC_UNICODE_H
#define LANGUAGE_BASIC_UNICODE_H

#include "language/Basic/Toolchain.h"
#include "toolchain/ADT/StringRef.h"

namespace language {
namespace unicode {

StringRef extractFirstExtendedGraphemeCluster(StringRef S);

static inline bool isSingleExtendedGraphemeCluster(StringRef S) {
  StringRef First = extractFirstExtendedGraphemeCluster(S);
  if (First.empty())
    return false;
  return First == S;
}

enum class GraphemeClusterBreakProperty : uint8_t {
  Other,
  CR,
  LF,
  Control,
  Extend,
  Regional_Indicator,
  Prepend,
  SpacingMark,
  L,
  V,
  T,
  LV,
  LVT,
};

/// Extended grapheme cluster boundary rules, represented as a matrix.  Indexed
/// by first code point, then by second code point in least-significant-bit
/// order.  A set bit means that a boundary is prohibited between two code
/// points.
extern const uint16_t ExtendedGraphemeClusterNoBoundaryRulesMatrix[];

/// Returns the value of the Grapheme_Cluster_Break property for a given code
/// point.
GraphemeClusterBreakProperty getGraphemeClusterBreakProperty(uint32_t C);

/// Determine if there is an extended grapheme cluster boundary between code
/// points with given Grapheme_Cluster_Break property values.
static inline bool
isExtendedGraphemeClusterBoundary(GraphemeClusterBreakProperty GCB1,
                                  GraphemeClusterBreakProperty GCB2) {
  auto RuleRow =
      ExtendedGraphemeClusterNoBoundaryRulesMatrix[static_cast<unsigned>(GCB1)];
  return !(RuleRow & (1 << static_cast<unsigned>(GCB2)));
}

bool isSingleUnicodeScalar(StringRef S);

unsigned extractFirstUnicodeScalar(StringRef S);

/// Returns true if \p S does not contain any ill-formed subsequences. This does
/// not check whether all of the characters in it are actually allocated or
/// used correctly; it just checks that every byte can be grouped into a code
/// unit (Unicode scalar).
bool isWellFormedUTF8(StringRef S);

/// Replaces any ill-formed subsequences with `u8"\ufffd"`.
std::string sanitizeUTF8(StringRef Text);

} // end namespace unicode
} // end namespace language

#endif // LANGUAGE_BASIC_UNICODE_H
