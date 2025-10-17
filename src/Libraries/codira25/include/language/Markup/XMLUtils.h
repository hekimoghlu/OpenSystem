/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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

//===--- XMLUtils.h - Various XML utility routines --------------*- C++ -*-===//
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

#ifndef LANGUAGE_MARKUP_XML_UTILS_H
#define LANGUAGE_MARKUP_XML_UTILS_H

#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/raw_ostream.h"

namespace language {
namespace markup {

// FIXME: copied from Clang's
// CommentASTToXMLConverter::appendToResultWithXMLEscaping
static inline void appendWithXMLEscaping(raw_ostream &OS, StringRef S) {
  auto Start = S.begin(), Cursor = Start, End = S.end();
  for (; Cursor != End; ++Cursor) {
    switch (*Cursor) {
    case '&':
      OS.write(Start, Cursor - Start);
      OS << "&amp;";
      break;
    case '<':
      OS.write(Start, Cursor - Start);
      OS << "&lt;";
      break;
    case '>':
      OS.write(Start, Cursor - Start);
      OS << "&gt;";
      break;
    case '"':
      OS.write(Start, Cursor - Start);
      OS << "&quot;";
      break;
    case '\'':
      OS.write(Start, Cursor - Start);
      OS << "&apos;";
      break;
    default:
      continue;
    }
    Start = Cursor + 1;
  }
  OS.write(Start, Cursor - Start);
}

// FIXME: copied from Clang's
// CommentASTToXMLConverter::appendToResultWithCDATAEscaping
static inline void appendWithCDATAEscaping(raw_ostream &OS, StringRef S) {
  if (S.empty())
    return;

  OS << "<![CDATA[";
  while (!S.empty()) {
    size_t Pos = S.find("]]>");
    if (Pos == 0) {
      OS << "]]]]><![CDATA[>";
      S = S.drop_front(3);
      continue;
    }
    if (Pos == StringRef::npos)
      Pos = S.size();

    OS << S.substr(0, Pos);

    S = S.drop_front(Pos);
  }
  OS << "]]>";
}

} // namespace markup
} // namespace language

#endif // LANGUAGE_MARKUP_XML_UTILS_H

