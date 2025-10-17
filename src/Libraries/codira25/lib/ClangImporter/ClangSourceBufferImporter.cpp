/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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

//===--- ClangSourceBufferImporter.cpp - Map Clang buffers to Codira -------===//
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

#include "ClangSourceBufferImporter.h"
#include "language/Basic/SourceManager.h"
#include "language/Core/Basic/SourceManager.h"
#include "toolchain/Support/MemoryBuffer.h"

using namespace language;
using namespace language::importer;

static SourceLoc findEndOfLine(SourceManager &SM, SourceLoc loc,
                               unsigned bufferID) {
  CharSourceRange entireBuffer = SM.getRangeForBuffer(bufferID);
  CharSourceRange rangeFromLoc{SM, loc, entireBuffer.getEnd()};
  StringRef textFromLoc = SM.extractText(rangeFromLoc);
  size_t newlineOffset = textFromLoc.find_first_of({"\r\n\0", 3});
  if (newlineOffset == StringRef::npos)
    return entireBuffer.getEnd();
  return loc.getAdvancedLoc(newlineOffset);
}

SourceLoc ClangSourceBufferImporter::resolveSourceLocation(
    const language::Core::SourceManager &clangSrcMgr,
    language::Core::SourceLocation clangLoc) {
  SourceLoc loc;

  clangLoc = clangSrcMgr.getFileLoc(clangLoc);
  auto decomposedLoc = clangSrcMgr.getDecomposedLoc(clangLoc);
  if (decomposedLoc.first.isInvalid())
    return loc;

  auto clangFileID = decomposedLoc.first;
  auto buffer = clangSrcMgr.getBufferOrFake(clangFileID);
  unsigned mirrorID;

  auto mirrorIter = mirroredBuffers.find(buffer.getBufferStart());
  if (mirrorIter != mirroredBuffers.end()) {
    mirrorID = mirrorIter->second;
  } else {
    std::unique_ptr<toolchain::MemoryBuffer> mirrorBuffer{
      toolchain::MemoryBuffer::getMemBuffer(buffer.getBuffer(),
                                       buffer.getBufferIdentifier(),
                                       /*RequiresNullTerminator=*/true)
    };
    mirrorID = languageSourceManager.addNewSourceBuffer(std::move(mirrorBuffer));
    mirroredBuffers[buffer.getBufferStart()] = mirrorID;
  }
  loc = languageSourceManager.getLocForOffset(mirrorID, decomposedLoc.second);

  auto presumedLoc = clangSrcMgr.getPresumedLoc(clangLoc);
  if (!presumedLoc.getFilename())
    return loc;
  if (presumedLoc.getLine() == 0)
    return SourceLoc();

  unsigned bufferLineNumber =
    clangSrcMgr.getLineNumber(decomposedLoc.first, decomposedLoc.second);

  StringRef presumedFile = presumedLoc.getFilename();
  SourceLoc startOfLine = loc.getAdvancedLoc(-presumedLoc.getColumn() + 1);

  // FIXME: Virtual files can't actually model the EOF position correctly, so
  // if this virtual file would start at EOF, just hope the physical location
  // will do.
  if (startOfLine != languageSourceManager.getRangeForBuffer(mirrorID).getEnd()) {
    bool isNewVirtualFile = languageSourceManager.openVirtualFile(
        startOfLine, presumedFile, presumedLoc.getLine() - bufferLineNumber);
    if (isNewVirtualFile) {
      SourceLoc endOfLine = findEndOfLine(languageSourceManager, loc, mirrorID);
      languageSourceManager.closeVirtualFile(endOfLine);
    }
  }

  using SourceManagerRef = toolchain::IntrusiveRefCntPtr<const language::Core::SourceManager>;
  auto iter = std::lower_bound(sourceManagersWithDiagnostics.begin(),
                               sourceManagersWithDiagnostics.end(),
                               &clangSrcMgr,
                               [](const SourceManagerRef &inArray,
                                  const language::Core::SourceManager *toInsert) {
    return std::less<const language::Core::SourceManager *>()(inArray.get(), toInsert);
  });
  if (iter == sourceManagersWithDiagnostics.end() ||
      iter->get() != &clangSrcMgr) {
    sourceManagersWithDiagnostics.insert(iter, &clangSrcMgr);
  }

  return loc;
}
