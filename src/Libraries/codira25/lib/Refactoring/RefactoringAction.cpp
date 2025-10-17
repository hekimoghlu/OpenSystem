/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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

//===----------------------------------------------------------------------===//
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

#include "RefactoringActions.h"

using namespace language::refactoring;

/// Get the source file that corresponds to the given buffer.
SourceFile *getContainingFile(ModuleDecl *M, RangeConfig Range) {
  auto &SM = M->getASTContext().SourceMgr;
  // TODO: We should add an ID -> SourceFile mapping.
  return M->getSourceFileContainingLocation(
      SM.getRangeForBuffer(Range.BufferID).getStart());
}

RefactoringAction::RefactoringAction(ModuleDecl *MD, RefactoringOptions &Opts,
                                     SourceEditConsumer &EditConsumer,
                                     DiagnosticConsumer &DiagConsumer)
    : MD(MD), TheFile(getContainingFile(MD, Opts.Range)),
      EditConsumer(EditConsumer), Ctx(MD->getASTContext()),
      SM(MD->getASTContext().SourceMgr), DiagEngine(SM),
      StartLoc(Lexer::getLocForStartOfToken(SM, Opts.Range.getStart(SM))),
      PreferredName(Opts.PreferredName) {
  DiagEngine.addConsumer(DiagConsumer);
}

TokenBasedRefactoringAction::TokenBasedRefactoringAction(
    ModuleDecl *MD, RefactoringOptions &Opts, SourceEditConsumer &EditConsumer,
    DiagnosticConsumer &DiagConsumer)
    : RefactoringAction(MD, Opts, EditConsumer, DiagConsumer) {
  // Resolve the sema token and save it for later use.
  CursorInfo =
      evaluateOrDefault(TheFile->getASTContext().evaluator,
                        CursorInfoRequest{CursorInfoOwner(TheFile, StartLoc)},
                        new ResolvedCursorInfo());
}

RangeBasedRefactoringAction::RangeBasedRefactoringAction(
    ModuleDecl *MD, RefactoringOptions &Opts, SourceEditConsumer &EditConsumer,
    DiagnosticConsumer &DiagConsumer)
    : RefactoringAction(MD, Opts, EditConsumer, DiagConsumer),
      RangeInfo(evaluateOrDefault(
          MD->getASTContext().evaluator,
          RangeInfoRequest(RangeInfoOwner(TheFile, Opts.Range.getStart(SM),
                                          Opts.Range.getEnd(SM))),
          ResolvedRangeInfo())) {}
