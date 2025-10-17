/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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

//===--- PrimitiveParsing.cpp - Primitive parsing routines ----------------===//
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

#include "language/Basic/Assertions.h"
#include "language/Basic/PrimitiveParsing.h"
#include "toolchain/ADT/SmallVector.h"

using namespace toolchain;

unsigned language::measureNewline(const char *BufferPtr, const char *BufferEnd) {
  if (BufferPtr == BufferEnd)
    return 0;

  if (*BufferPtr == '\n')
    return 1;

  assert(*BufferPtr == '\r');
  unsigned Bytes = 1;
  if (BufferPtr != BufferEnd && *BufferPtr == '\n')
    ++Bytes;
  return Bytes;
}

void
language::trimLeadingWhitespaceFromLines(StringRef RawText,
                                      unsigned WhitespaceToTrim,
                                      SmallVectorImpl<StringRef> &OutLines) {
  SmallVector<StringRef, 8> Lines;

  bool IsFirstLine = true;

  while (!RawText.empty()) {
    size_t Pos = RawText.find_first_of("\n\r");
    if (Pos == StringRef::npos)
      Pos = RawText.size();

    StringRef Line = RawText.substr(0, Pos);
    Lines.push_back(Line);
    if (!IsFirstLine) {
      size_t NonWhitespacePos = RawText.find_first_not_of(' ');
      if (NonWhitespacePos != StringRef::npos)
        WhitespaceToTrim =
            std::min(WhitespaceToTrim,
                     static_cast<unsigned>(NonWhitespacePos));
    }
    IsFirstLine = false;

    RawText = RawText.drop_front(Pos);
    unsigned NewlineBytes = measureNewline(RawText);
    RawText = RawText.drop_front(NewlineBytes);
  }

  IsFirstLine = true;
  for (auto &Line : Lines) {
    if (!IsFirstLine) {
      Line = Line.drop_front(WhitespaceToTrim);
    }
    IsFirstLine = false;
  }

  OutLines.append(Lines.begin(), Lines.end());
}
