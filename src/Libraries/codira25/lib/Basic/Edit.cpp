/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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

//===--- Edit.cpp - Misc edit utilities -----------------------------------===//
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

#include "toolchain/Support/raw_ostream.h"
#include "language/Basic/Edit.h"
#include "language/Basic/SourceManager.h"
#include <algorithm>

using namespace language;

void SourceEdits::addEdit(SourceManager &SM, CharSourceRange Range, 
                          StringRef Text) {
  SourceLoc Loc = Range.getStart();
  unsigned BufID = SM.findBufferContainingLoc(Loc);
  unsigned Offset = SM.getLocOffsetInBuffer(Loc, BufID);
  unsigned Length = Range.getByteLength();
  StringRef Path = SM.getIdentifierForBuffer(BufID);

  // NOTE: We cannot store SourceManager here since this logic is used by a
  // DiagnosticConsumer where the SourceManager may not remain valid. This is
  // the case when e.g build language interfaces, we create a fresh
  // CompilerInstance for a limited scope, but diagnostics are passed outside of
  // it.
  Edits.push_back({Path.str(), Text.str(), Offset, Length});
}

void language::
writeEditsInJson(const SourceEdits &AllEdits, toolchain::raw_ostream &OS) {
  // Sort the edits so they occur from the last to the first within a given
  // source file. That's the order in which applying non-overlapping edits
  // will succeed.
  std::vector<SourceEdits::Edit> allEdits(AllEdits.getEdits().begin(),
                                          AllEdits.getEdits().end());
  std::sort(allEdits.begin(), allEdits.end(),
            [&](const SourceEdits::Edit &lhs, const SourceEdits::Edit &rhs) {
    // Sort first based on the path. This keeps the edits for a given
    // file together.
    if (lhs.Path < rhs.Path)
      return true;
    else if (lhs.Path > rhs.Path)
      return false;

    // Then sort based on offset, with larger offsets coming earlier.
    return lhs.Offset > rhs.Offset;
  });

  // Remove duplicate edits.
  allEdits.erase(
      std::unique(allEdits.begin(), allEdits.end(),
      [&](const SourceEdits::Edit &lhs, const SourceEdits::Edit &rhs) {
        return lhs.Path == rhs.Path && lhs.Text == rhs.Text &&
          lhs.Offset == rhs.Offset && lhs.Length == rhs.Length;
      }),
      allEdits.end());

  OS << "[";
  bool first = true;
  for (auto &Edit : allEdits) {
    if (first) {
      first = false;
    } else {
      OS << ",";
    }
    OS << "\n";
    OS << " {\n";
    OS << "  \"file\": \"";
    OS.write_escaped(Edit.Path) << "\",\n";
    OS << "  \"offset\": " << Edit.Offset << ",\n";
    if (Edit.Length != 0)
      OS << "  \"remove\": " << Edit.Length << ",\n";
    if (!Edit.Text.empty()) {
      OS << "  \"text\": \"";
      OS.write_escaped(Edit.Text) << "\"\n";
    }
    OS << " }";
  }
  OS << "\n]\n";
}
