/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

//===--- PrimitiveParsing.h - Primitive parsing routines --------*- C++ -*-===//
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
///
/// \file
/// Primitive parsing routines useful in various places in the compiler.
///
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BASIC_PRIMITIVEPARSING_H
#define LANGUAGE_BASIC_PRIMITIVEPARSING_H

#include "toolchain/ADT/StringRef.h"
#include "language/Basic/Toolchain.h"

namespace language {

unsigned measureNewline(const char *BufferPtr, const char *BufferEnd);

static inline unsigned measureNewline(StringRef S) {
  return measureNewline(S.data(), S.data() + S.size());
}

static inline bool startsWithNewline(StringRef S) {
  return S.starts_with("\n") || S.starts_with("\r\n");
}

/// Breaks a given string to lines and trims leading whitespace from them.
void trimLeadingWhitespaceFromLines(StringRef Text, unsigned WhitespaceToTrim,
                                    SmallVectorImpl<StringRef> &Lines);

static inline void splitIntoLines(StringRef Text,
                                  SmallVectorImpl<StringRef> &Lines) {
  trimLeadingWhitespaceFromLines(Text, 0, Lines);
}

} // end namespace language

#endif // LANGUAGE_BASIC_PRIMITIVEPARSING_H

