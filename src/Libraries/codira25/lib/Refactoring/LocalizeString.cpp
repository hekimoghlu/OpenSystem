/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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

static Expr *findLocalizeTarget(ResolvedCursorInfoPtr CursorInfo) {
  auto ExprStartInfo = dyn_cast<ResolvedExprStartCursorInfo>(CursorInfo);
  if (!ExprStartInfo)
    return nullptr;
  struct StringLiteralFinder : public SourceEntityWalker {
    SourceLoc StartLoc;
    Expr *Target;
    StringLiteralFinder(SourceLoc StartLoc)
        : StartLoc(StartLoc), Target(nullptr) {}
    bool walkToExprPre(Expr *E) override {
      if (E->getStartLoc() != StartLoc)
        return false;
      if (E->getKind() == ExprKind::InterpolatedStringLiteral)
        return false;
      if (E->getKind() == ExprKind::StringLiteral) {
        Target = E;
        return false;
      }
      return true;
    }
  } Walker(ExprStartInfo->getTrailingExpr()->getStartLoc());
  Walker.walk(ExprStartInfo->getTrailingExpr());
  return Walker.Target;
}

bool RefactoringActionLocalizeString::isApplicable(ResolvedCursorInfoPtr Tok,
                                                   DiagnosticEngine &Diag) {
  return findLocalizeTarget(Tok);
}

bool RefactoringActionLocalizeString::performChange() {
  Expr *Target = findLocalizeTarget(CursorInfo);
  if (!Target)
    return true;
  EditConsumer.accept(SM, Target->getStartLoc(), "NSLocalizedString(");
  EditConsumer.insertAfter(SM, Target->getEndLoc(), ", comment: \"\")");
  return false;
}
