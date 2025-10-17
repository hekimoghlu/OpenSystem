/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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

//===- toolchain/Support/Unicode.h - Unicode character properties  -*- C++ -*-=====//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
// This file defines functions that allow querying certain properties of Unicode
// characters.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_UNICODE_H
#define LLVM_SUPPORT_UNICODE_H

#include <IndexStoreDB_LLVMSupport/toolchain_Config_indexstoredb-prefix.h>

namespace toolchain {
class StringRef;

namespace sys {
namespace unicode {

enum ColumnWidthErrors {
  ErrorInvalidUTF8 = -2,
  ErrorNonPrintableCharacter = -1
};

/// Determines if a character is likely to be displayed correctly on the
/// terminal. Exact implementation would have to depend on the specific
/// terminal, so we define the semantic that should be suitable for generic case
/// of a terminal capable to output Unicode characters.
///
/// All characters from the Unicode code point range are considered printable
/// except for:
///   * C0 and C1 control character ranges;
///   * default ignorable code points as per 5.21 of
///     http://www.unicode.org/versions/Unicode6.2.0/UnicodeStandard-6.2.pdf
///     except for U+00AD SOFT HYPHEN, as it's actually displayed on most
///     terminals;
///   * format characters (category = Cf);
///   * surrogates (category = Cs);
///   * unassigned characters (category = Cn).
/// \return true if the character is considered printable.
bool isPrintable(int UCS);

/// Gets the number of positions the UTF8-encoded \p Text is likely to occupy
/// when output on a terminal ("character width"). This depends on the
/// implementation of the terminal, and there's no standard definition of
/// character width.
///
/// The implementation defines it in a way that is expected to be compatible
/// with a generic Unicode-capable terminal.
///
/// \return Character width:
///   * ErrorNonPrintableCharacter (-1) if \p Text contains non-printable
///     characters (as identified by isPrintable);
///   * 0 for each non-spacing and enclosing combining mark;
///   * 2 for each CJK character excluding halfwidth forms;
///   * 1 for each of the remaining characters.
int columnWidthUTF8(StringRef Text);

/// Fold input unicode character according the Simple unicode case folding
/// rules.
int foldCharSimple(int C);

} // namespace unicode
} // namespace sys
} // namespace toolchain

#endif
