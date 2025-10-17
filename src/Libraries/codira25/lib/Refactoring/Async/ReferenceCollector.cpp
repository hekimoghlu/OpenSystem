/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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

#include "AsyncRefactoring.h"

using namespace language;
using namespace language::refactoring::asyncrefactorings;

void ReferenceCollector::collect(ASTNode Target, BraceStmt *Scope,
                                 SourceFile &SF,
                                 toolchain::DenseSet<const Decl *> &Decls) {
  ReferenceCollector Collector(Target, &SF.getASTContext().SourceMgr, Decls);
  if (Scope)
    Collector.walk(Scope);
  else
    Collector.walk(SF);
}

bool ReferenceCollector::walkToDeclPre(Decl *D, CharSourceRange Range) {
  // Bit of a hack, include all contexts so they're never renamed (seems worse
  // to rename a class/function than it does a variable). Again, an
  // over-approximation, but hopefully doesn't come up too often.
  if (isa<DeclContext>(D) && !D->isImplicit()) {
    ReferencedDecls.insert(D);
  }

  if (AfterTarget && !D->isImplicit()) {
    DeclaredDecls.insert(D);
  } else if (D == Target.dyn_cast<Decl *>()) {
    AfterTarget = true;
  }
  return shouldWalkInto(D->getSourceRange());
}

bool ReferenceCollector::walkToExprPre(Expr *E) {
  if (AfterTarget && !E->isImplicit()) {
    if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      if (auto *D = DRE->getDecl()) {
        // Only care about references that aren't declared, as seen decls will
        // be renamed (if necessary) during the refactoring.
        if (!D->isImplicit() && !DeclaredDecls.count(D)) {
          ReferencedDecls.insert(D);

          // Also add the async alternative of a function to prevent
          // collisions if a call is replaced with the alternative.
          if (auto *AFD = dyn_cast<AbstractFunctionDecl>(D)) {
            if (auto *Alternative = AFD->getAsyncAlternative())
              ReferencedDecls.insert(Alternative);
          }
        }
      }
    }
  } else if (E == Target.dyn_cast<Expr *>()) {
    AfterTarget = true;
  }
  return shouldWalkInto(E->getSourceRange());
}

bool ReferenceCollector::walkToStmtPre(Stmt *S) {
  if (S == Target.dyn_cast<Stmt *>())
    AfterTarget = true;
  return shouldWalkInto(S->getSourceRange());
}

bool ReferenceCollector::walkToPatternPre(Pattern *P) {
  if (P == Target.dyn_cast<Pattern *>())
    AfterTarget = true;
  return shouldWalkInto(P->getSourceRange());
}

bool ReferenceCollector::shouldWalkInto(SourceRange Range) {
  return AfterTarget ||
         (SM && SM->rangeContainsTokenLoc(Range, Target.getStartLoc()));
}
