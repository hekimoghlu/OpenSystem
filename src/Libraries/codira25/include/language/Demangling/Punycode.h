/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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

//===--- Punycode.h - UTF-8 to Punycode transcoding -------------*- C++ -*-===//
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
// These functions implement a variant of the Punycode algorithm from RFC3492,
// originally designed for encoding international domain names, for the purpose
// of encoding Codira identifiers into mangled symbol names. This version differs
// from RFC3492 in the following respects:
// - '_' is used as the encoding delimiter instead of '-'.
// - Encoding digits are represented using [a-zA-J] instead of [a-z0-9], because
//   symbol names are case-sensitive, and Codira mangled identifiers cannot begin
//   with a digit.
// - Optionally, non-symbol ASCII characters (characters except [$_a-zA-Z0-9])
//   are mapped to the code range 0xD800 - 0xD880 and are also encoded like
//   non-ASCII unicode characters.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_DEMANGLING_PUNYCODE_H
#define LANGUAGE_DEMANGLING_PUNYCODE_H

#include "toolchain/ADT/StringRef.h"
#include "language/Demangling/NamespaceMacros.h"
#include <vector>
#include <cstdint>

namespace language {
namespace Punycode {
LANGUAGE_BEGIN_INLINE_NAMESPACE

using toolchain::StringRef;

/// Encodes a sequence of code points into Punycode.
///
/// Returns false if input contains surrogate code points.
bool encodePunycode(const std::vector<uint32_t> &InputCodePoints,
                    std::string &OutPunycode);

/// Decodes a Punycode string into a sequence of Unicode scalars.
///
/// Returns false if decoding failed.
bool decodePunycode(StringRef InputPunycode,
                    std::vector<uint32_t> &OutCodePoints);

/// Encodes an UTF8 string into Punycode.
///
/// If \p mapNonSymbolChars is true, non-symbol ASCII characters (characters
/// except [$_a-zA-Z0-9]) are also encoded like non-ASCII unicode characters.
/// Returns false if \p InputUTF8 contains surrogate code points.
bool encodePunycodeUTF8(StringRef InputUTF8, std::string &OutPunycode,
                        bool mapNonSymbolChars = false);

bool decodePunycodeUTF8(StringRef InputPunycode, std::string &OutUTF8);

LANGUAGE_END_INLINE_NAMESPACE
} // end namespace Punycode
} // end namespace language

#endif // LANGUAGE_DEMANGLING_PUNYCODE_H

