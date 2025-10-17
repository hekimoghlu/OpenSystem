/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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

//===--- LineList.h - Data structures for Markup parsing --------*- C++ -*-===//
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

#ifndef LANGUAGE_MARKUP_LINELIST_H
#define LANGUAGE_MARKUP_LINELIST_H

#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/StringRef.h"
#include "language/Basic/SourceLoc.h"

namespace language {
namespace markup {

class MarkupContext;

/// Returns the amount of indentation on the line and
/// the length of the line's length.
size_t measureIndentation(StringRef Text);

/// Represents a substring of a single line of source text.
struct Line {
  StringRef Text;
  language::SourceRange Range;
  size_t FirstNonspaceOffset;
public:
  Line(StringRef Text, language::SourceRange Range) : Text(Text), Range(Range),
      FirstNonspaceOffset(measureIndentation(Text)) {}

  void drop_front(size_t Amount) {
    Text = Text.drop_front(Amount);
  }
};

/// A possibly non-contiguous view into a source buffer.
class LineList {
  MutableArrayRef<Line> Lines;
public:
  LineList(MutableArrayRef<Line> Lines) : Lines(Lines) {}
  LineList() = default;

  std::string str() const;

  ArrayRef<Line> getLines() const {
    return Lines;
  }

  /// Creates a LineList from a box selection of text.
  ///
  /// \param MC MarkupContext used for allocations
  /// \param StartLine 0-based start line
  /// \param EndLine 0-based end line (1 off the end)
  /// \param StartColumn 0-based start column
  /// \param EndColumn 0-based end column (1 off the end)
  /// \returns a new LineList with the selected start and end lines/columns.
  LineList subListWithRange(MarkupContext &MC, size_t StartLine, size_t EndLine,
                            size_t StartColumn, size_t EndColumn) const;
};

class LineListBuilder {
  std::vector<Line> Lines;
  MarkupContext &Context;
public:
  LineListBuilder(MarkupContext &Context) : Context(Context) {}

  void addLine(StringRef Text, language::SourceRange Range);
  LineList takeLineList() const;
};

} // namespace markup
} // namespace language

#endif // LANGUAGE_MARKUP_LINELIST_H
