/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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

//===--- PersistentParserState.cpp - Parser State Implementation ----------===//
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
//  This file implements parser state persistent across multiple parses.
//
//===----------------------------------------------------------------------===//

#include "language/AST/ASTContext.h"
#include "language/AST/Decl.h"
#include "language/AST/Expr.h"
#include "language/Basic/Assertions.h"
#include "language/Basic/SourceManager.h"
#include "language/Parse/PersistentParserState.h"

using namespace language;

PersistentParserState::PersistentParserState() { }

PersistentParserState::~PersistentParserState() { }

void PersistentParserState::setIDEInspectionDelayedDeclState(
    SourceManager &SM, unsigned BufferID, IDEInspectionDelayedDeclKind Kind,
    DeclContext *ParentContext, SourceRange BodyRange, SourceLoc PreviousLoc) {
  assert(!IDEInspectionDelayedDeclStat.get() &&
         "only one decl can be delayed for code completion");
  unsigned startOffset = SM.getLocOffsetInBuffer(BodyRange.Start, BufferID);
  unsigned endOffset = SM.getLocOffsetInBuffer(BodyRange.End, BufferID);
  unsigned prevOffset = ~0U;
  if (PreviousLoc.isValid())
    prevOffset = SM.getLocOffsetInBuffer(PreviousLoc, BufferID);

  IDEInspectionDelayedDeclStat.reset(new IDEInspectionDelayedDeclState(
      Kind, ParentContext, startOffset, endOffset, prevOffset));
}

void PersistentParserState::restoreIDEInspectionDelayedDeclState(
    const IDEInspectionDelayedDeclState &other) {
  IDEInspectionDelayedDeclStat.reset(new IDEInspectionDelayedDeclState(
      other.Kind, other.ParentContext, other.StartOffset, other.EndOffset,
      other.PrevOffset));
}
