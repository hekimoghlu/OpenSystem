/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 8, 2024.
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

namespace {
// A SingleDecl range may not include all decls actually declared in that range:
// a var decl has accessors that aren't included. This will find those missing
// decls.
class FindAllSubDecls : public SourceEntityWalker {
  SmallPtrSetImpl<Decl *> &Found;

public:
  FindAllSubDecls(SmallPtrSetImpl<Decl *> &found) : Found(found) {}

  bool walkToDeclPre(Decl *D, CharSourceRange range) override {
    // Record this Decl, and skip its contents if we've already touched it.
    if (!Found.insert(D).second)
      return false;

    if (auto ASD = dyn_cast<AbstractStorageDecl>(D)) {
      ASD->visitParsedAccessors(
          [&](AccessorDecl *accessor) { Found.insert(accessor); });
    }
    return true;
  }
};
} // namespace

bool RefactoringActionReplaceBodiesWithFatalError::isApplicable(
    const ResolvedRangeInfo &Info, DiagnosticEngine &Diag) {
  switch (Info.Kind) {
  case RangeKind::SingleDecl:
  case RangeKind::MultiTypeMemberDecl: {
    SmallPtrSet<Decl *, 16> Found;
    for (auto decl : Info.DeclaredDecls) {
      FindAllSubDecls(Found).walk(decl.VD);
    }
    for (auto decl : Found) {
      auto AFD = dyn_cast<AbstractFunctionDecl>(decl);
      if (AFD && !AFD->isImplicit())
        return true;
    }

    return false;
  }
  case RangeKind::SingleExpression:
  case RangeKind::PartOfExpression:
  case RangeKind::SingleStatement:
  case RangeKind::MultiStatement:
  case RangeKind::Invalid:
    return false;
  }
  toolchain_unreachable("unhandled kind");
}

bool RefactoringActionReplaceBodiesWithFatalError::performChange() {
  const StringRef replacement = "{\nfatalError()\n}";
  SmallPtrSet<Decl *, 16> Found;
  for (auto decl : RangeInfo.DeclaredDecls) {
    FindAllSubDecls(Found).walk(decl.VD);
  }
  for (auto decl : Found) {
    auto AFD = dyn_cast<AbstractFunctionDecl>(decl);
    if (!AFD || AFD->isImplicit())
      continue;

    auto range = AFD->getBodySourceRange();
    // If we're in replacement mode (i.e. have an edit consumer), we can
    // rewrite the function body.
    auto charRange = Lexer::getCharSourceRangeFromSourceRange(SM, range);
    EditConsumer.accept(SM, charRange, replacement);
  }
  return false;
}
