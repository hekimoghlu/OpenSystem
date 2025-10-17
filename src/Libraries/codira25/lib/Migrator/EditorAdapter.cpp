/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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

//===--- EditorAdapter.cpp ------------------------------------------------===//
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

#include "language/Basic/SourceLoc.h"
#include "language/Basic/SourceManager.h"
#include "language/Migrator/EditorAdapter.h"
#include "language/Migrator/Replacement.h"
#include "language/Parse/Lexer.h"
#include "language/Core/Basic/SourceManager.h"

using namespace language;
using namespace language::migrator;

std::pair<unsigned, unsigned>
EditorAdapter::getLocInfo(language::SourceLoc Loc) const {
  auto CodiraBufferID = CodiraSrcMgr.findBufferContainingLoc(Loc);
  auto Offset = CodiraSrcMgr.getLocOffsetInBuffer(Loc, CodiraBufferID);
  return { CodiraBufferID, Offset };
}

bool
EditorAdapter::cacheReplacement(CharSourceRange Range, StringRef Text) const {
  if (!CacheEnabled)
    return false;
  unsigned CodiraBufferID, Offset;
  std::tie(CodiraBufferID, Offset) = getLocInfo(Range.getStart());
  Replacement R{Offset, Range.getByteLength(), Text.str()};
  if (Replacements.count(R)) {
    return true;
  } else {
    Replacements.insert(R);
  }
  return false;
}

language::Core::FileID
EditorAdapter::getClangFileIDForCodiraBufferID(unsigned BufferID) const {
  /// Check if we already have a mapping for this BufferID.
  auto Found = CodiraToClangBufferMap.find(BufferID);
  if (Found != CodiraToClangBufferMap.end()) {
    return Found->getSecond();
  }

  // If we don't copy the corresponding buffer's text into a new buffer
  // that the ClangSrcMgr can understand.
  auto Text = CodiraSrcMgr.getEntireTextForBuffer(BufferID);
  auto NewBuffer = toolchain::MemoryBuffer::getMemBufferCopy(Text);
  auto NewFileID = ClangSrcMgr.createFileID(std::move(NewBuffer));

  CodiraToClangBufferMap.insert({BufferID, NewFileID});

  return NewFileID;
}

language::Core::SourceLocation
EditorAdapter::translateSourceLoc(SourceLoc CodiraLoc) const {
  unsigned CodiraBufferID, Offset;
  std::tie(CodiraBufferID, Offset) = getLocInfo(CodiraLoc);

  auto ClangFileID = getClangFileIDForCodiraBufferID(CodiraBufferID);
  return ClangSrcMgr.getLocForStartOfFile(ClangFileID).getLocWithOffset(Offset);
}

language::Core::SourceRange
EditorAdapter::translateSourceRange(SourceRange CodiraSourceRange) const {
  auto Start = translateSourceLoc(CodiraSourceRange.Start);
  auto End = translateSourceLoc(CodiraSourceRange.End);
  return language::Core::SourceRange { Start, End };
}

language::Core::CharSourceRange EditorAdapter::
translateCharSourceRange(CharSourceRange CodiraSourceSourceRange) const {
  auto ClangStartLoc = translateSourceLoc(CodiraSourceSourceRange.getStart());
  auto ClangEndLoc = translateSourceLoc(CodiraSourceSourceRange.getEnd());
  return language::Core::CharSourceRange::getCharRange(ClangStartLoc, ClangEndLoc);
}

bool EditorAdapter::insert(SourceLoc Loc, StringRef Text, bool AfterToken,
                           bool BeforePreviousInsertions) {
  // We don't have tokens on the clang side, so handle AfterToken in Codira
  if (AfterToken)
    Loc = Lexer::getLocForEndOfToken(CodiraSrcMgr, Loc);

  if (cacheReplacement(CharSourceRange { Loc, 0 }, Text)) {
    return true;
  }

  auto ClangLoc = translateSourceLoc(Loc);
  return Edits.insert(ClangLoc, Text, /*AfterToken=*/false, BeforePreviousInsertions);
}

bool EditorAdapter::insertFromRange(SourceLoc Loc, CharSourceRange Range,
                                    bool AfterToken,
                                    bool BeforePreviousInsertions) {
  // We don't have tokens on the clang side, so handle AfterToken in Codira
  if (AfterToken)
    Loc = Lexer::getLocForEndOfToken(CodiraSrcMgr, Loc);

  if (cacheReplacement(CharSourceRange { Loc, 0 },
                       CodiraSrcMgr.extractText(Range))) {
    return true;
  }

  auto ClangLoc = translateSourceLoc(Loc);
  auto ClangCharRange = translateCharSourceRange(Range);

  return Edits.insertFromRange(ClangLoc, ClangCharRange, /*AfterToken=*/false,
                               BeforePreviousInsertions);
}

bool EditorAdapter::insertWrap(StringRef Before, CharSourceRange Range,
                               StringRef After) {
  auto ClangRange = translateCharSourceRange(Range);
  return Edits.insertWrap(Before, ClangRange, After);
}

bool EditorAdapter::remove(CharSourceRange Range) {
  if (cacheReplacement(Range, "")) {
    return true;
  }
  auto ClangRange = translateCharSourceRange(Range);
  return Edits.remove(ClangRange);
}

bool EditorAdapter::replace(CharSourceRange Range, StringRef Text) {
  if (cacheReplacement(Range, Text)) {
    return true;
  }

  auto ClangRange = translateCharSourceRange(Range);
  return Edits.replace(ClangRange, Text);
}

bool EditorAdapter::replaceWithInner(CharSourceRange Range,
                                     CharSourceRange InnerRange) {

  if (cacheReplacement(Range, CodiraSrcMgr.extractText(InnerRange))) {
    return true;
  }
  auto ClangRange = translateCharSourceRange(Range);
  auto ClangInnerRange = translateCharSourceRange(InnerRange);
  return Edits.replaceWithInner(ClangRange, ClangInnerRange);
}

bool EditorAdapter::replaceText(SourceLoc Loc, StringRef Text,
                                StringRef ReplacementText) {
  auto Range = Lexer::getCharSourceRangeFromSourceRange(CodiraSrcMgr,
    { Loc, Loc.getAdvancedLoc(Text.size())});
  if (cacheReplacement(Range, Text)) {
    return true;
  }

  auto ClangLoc = translateSourceLoc(Loc);
  return Edits.replaceText(ClangLoc, Text, ReplacementText);
}

bool EditorAdapter::insertFromRange(SourceLoc Loc, SourceRange TokenRange,
                     bool AfterToken,
                     bool BeforePreviousInsertions) {
  auto CharRange = Lexer::getCharSourceRangeFromSourceRange(CodiraSrcMgr, TokenRange);
  return insertFromRange(Loc, CharRange,
                         AfterToken, BeforePreviousInsertions);
}

bool EditorAdapter::insertWrap(StringRef Before, SourceRange TokenRange,
                               StringRef After) {
  auto CharRange = Lexer::getCharSourceRangeFromSourceRange(CodiraSrcMgr, TokenRange);
  return insertWrap(Before, CharRange, After);
}

bool EditorAdapter::remove(SourceLoc TokenLoc) {
  return remove(Lexer::getCharSourceRangeFromSourceRange(CodiraSrcMgr,
                                                         TokenLoc));
}

bool EditorAdapter::remove(SourceRange TokenRange) {
  auto CharRange = Lexer::getCharSourceRangeFromSourceRange(CodiraSrcMgr, TokenRange);
  return remove(CharRange);
}

bool EditorAdapter::replace(SourceRange TokenRange, StringRef Text) {
  auto CharRange = Lexer::getCharSourceRangeFromSourceRange(CodiraSrcMgr,TokenRange);
  return replace(CharRange, Text);
}

bool EditorAdapter::replaceWithInner(SourceRange TokenRange,
                                     SourceRange TokenInnerRange) {
  auto CharRange = Lexer::getCharSourceRangeFromSourceRange(CodiraSrcMgr, TokenRange);
  auto CharInnerRange = Lexer::getCharSourceRangeFromSourceRange(CodiraSrcMgr, TokenInnerRange);
  return replaceWithInner(CharRange, CharInnerRange);
}

bool EditorAdapter::replaceToken(SourceLoc TokenLoc, StringRef Text) {
  return replace(Lexer::getTokenAtLocation(CodiraSrcMgr, TokenLoc).getRange(),
    Text);
}
