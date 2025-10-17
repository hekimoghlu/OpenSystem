/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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

//===--- PersistentParserState.h - Parser State -----------------*- C++ -*-===//
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
// Parser state persistent across multiple parses.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_PARSE_PERSISTENTPARSERSTATE_H
#define LANGUAGE_PARSE_PERSISTENTPARSERSTATE_H

#include "language/Basic/SourceLoc.h"

namespace language {

class SourceFile;
class DeclContext;
class IterableDeclContext;

enum class IDEInspectionDelayedDeclKind {
  TopLevelCodeDecl,
  Decl,
  FunctionBody,
};

class IDEInspectionDelayedDeclState {
public:
  IDEInspectionDelayedDeclKind Kind;
  DeclContext *ParentContext;
  unsigned StartOffset;
  unsigned EndOffset;
  unsigned PrevOffset;

  IDEInspectionDelayedDeclState(IDEInspectionDelayedDeclKind Kind,
                                DeclContext *ParentContext,
                                unsigned StartOffset, unsigned EndOffset,
                                unsigned PrevOffset)
      : Kind(Kind), ParentContext(ParentContext), StartOffset(StartOffset),
        EndOffset(EndOffset), PrevOffset(PrevOffset) {}
};

/// Parser state persistent across multiple parses.
class PersistentParserState {
  std::unique_ptr<IDEInspectionDelayedDeclState> IDEInspectionDelayedDeclStat;

public:
  PersistentParserState();
  PersistentParserState(ASTContext &ctx) : PersistentParserState() { }
  ~PersistentParserState();

  void setIDEInspectionDelayedDeclState(SourceManager &SM, unsigned BufferID,
                                        IDEInspectionDelayedDeclKind Kind,
                                        DeclContext *ParentContext,
                                        SourceRange BodyRange,
                                        SourceLoc PreviousLoc);
  void restoreIDEInspectionDelayedDeclState(
      const IDEInspectionDelayedDeclState &other);

  bool hasIDEInspectionDelayedDeclState() const {
    return IDEInspectionDelayedDeclStat.get() != nullptr;
  }

  IDEInspectionDelayedDeclState &getIDEInspectionDelayedDeclState() {
    return *IDEInspectionDelayedDeclStat.get();
  }
  const IDEInspectionDelayedDeclState &
  getIDEInspectionDelayedDeclState() const {
    return *IDEInspectionDelayedDeclStat.get();
  }

  std::unique_ptr<IDEInspectionDelayedDeclState>
  takeIDEInspectionDelayedDeclState() {
    assert(hasIDEInspectionDelayedDeclState());
    return std::move(IDEInspectionDelayedDeclStat);
  }
};

} // end namespace language

#endif
