/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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

//===--- Indenting.h --------------------------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_INDENTING_H
#define LANGUAGE_INDENTING_H

namespace language {
namespace ide {

struct CodeFormatOptions {
public:
  bool UseTabs = false;
  bool IndentSwitchCase = false;
  unsigned IndentWidth = 4;
  unsigned TabWidth = 4;
};

/// Returns the offset (in bytes) to the start of \p LineIndex
size_t getOffsetOfLine(unsigned LineIndex, StringRef Text);

/// Returns the offset to the first Character. If \p Trim is true, the
///    first character is Non-WhiteSpace.
size_t getOffsetOfLine(unsigned LineIndex, StringRef Text, bool Trim);

/// Returns the Text on \p LineIndex, excluding Leading WS if \p Trim is
///   true.
StringRef getTextForLine(unsigned LineIndex, StringRef Text, bool Trim);

/// Returns the number of spaces at the beginning of \p LineIndex
/// or if indenting is done by Tabs, the number of Tabs * TabWidthp
size_t getExpandedIndentForLine(unsigned LineIndex, CodeFormatOptions Options,
                                StringRef Text);

class LineRange {
  unsigned StartLine;
  unsigned Length;

public:
  LineRange()
    :StartLine(0), Length(0) { }
  LineRange(unsigned StartLine, unsigned Length)
    :StartLine(StartLine), Length(Length) { }
  LineRange(const LineRange &Other)
    :StartLine(Other.StartLine), Length(Other.Length) { }

  bool isValid() const {
    return Length != 0;
  }

  unsigned startLine() const {
    return StartLine;
  }

  unsigned endLine() const {
    return isValid() ? StartLine + Length - 1 : 0;
  }

  unsigned lineCount() const {
    return Length;
  }

  void setRange(unsigned NewStartLine, unsigned NewLength) {
    StartLine = NewStartLine;
    Length = NewLength;
  }

  void extendToIncludeLine(unsigned Line) {
    if (!isValid()) {
      StartLine = Line;
      Length = 1;
    }
    else if (Line >= StartLine + Length) {
      Length = Line - StartLine + 1;
    }
  }

};

//===----------------------------------------------------------------------===//
// Reformat
//===----------------------------------------------------------------------===//
/// Request a reformatting of the \p Range, using \p Options to determine the
/// how the format should be applied to \p SF. \p SM is required to provide
/// an ASTContext and other helper data.
/// \returns a pair containing which line ranges where updated and a string
/// containing the applied edits.
std::pair<LineRange, std::string> reformat(LineRange Range,
                                           CodeFormatOptions Options,
                                           SourceManager &SM,
                                           SourceFile &SF);

} // namespace ide
} // namespace language

#endif // LANGUAGE_INDENTING_H
