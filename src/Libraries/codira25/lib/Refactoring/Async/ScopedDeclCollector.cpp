/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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

const ScopedDeclCollector::RefDeclsTy *
ScopedDeclCollector::getReferencedDecls(Stmt *Scope) const {
  auto Res = ReferencedDecls.find(Scope);
  if (Res == ReferencedDecls.end())
    return nullptr;
  return &Res->second;
}

bool ScopedDeclCollector::walkToDeclPre(Decl *D, CharSourceRange Range) {
  if (ScopeStack.empty() || D->isImplicit())
    return true;

  ScopeStack.back().DeclaredDecls.insert(D);
  if (isa<DeclContext>(D))
    (*ScopeStack.back().ReferencedDecls)[D] += 1;
  return true;
}

bool ScopedDeclCollector::walkToExprPre(Expr *E) {
  if (ScopeStack.empty())
    return true;

  if (!E->isImplicit()) {
    if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      if (auto *D = DRE->getDecl()) {
        // If we have a reference that isn't declared in the same scope,
        // increment the number of references to that decl.
        if (!D->isImplicit() && !ScopeStack.back().DeclaredDecls.count(D)) {
          (*ScopeStack.back().ReferencedDecls)[D] += 1;

          // Also add the async alternative of a function to prevent
          // collisions if a call is replaced with the alternative.
          if (auto *AFD = dyn_cast<AbstractFunctionDecl>(D)) {
            if (auto *Alternative = AFD->getAsyncAlternative())
              (*ScopeStack.back().ReferencedDecls)[Alternative] += 1;
          }
        }
      }
    }
  }
  return true;
}

bool ScopedDeclCollector::walkToStmtPre(Stmt *S) {
  // Purposely check \c BraceStmt here rather than \c startsNewScope.
  // References in the condition should be applied to the previous scope, not
  // the scope of that statement.
  if (isa<BraceStmt>(S))
    ScopeStack.emplace_back(&ReferencedDecls[S]);
  return true;
}

bool ScopedDeclCollector::walkToStmtPost(Stmt *S) {
  if (isa<BraceStmt>(S)) {
    size_t NumScopes = ScopeStack.size();
    if (NumScopes >= 2) {
      // Add any referenced decls to the parent scope that weren't declared
      // there.
      auto &ParentStack = ScopeStack[NumScopes - 2];
      for (auto DeclAndNumRefs : *ScopeStack.back().ReferencedDecls) {
        auto *D = DeclAndNumRefs.first;
        if (!ParentStack.DeclaredDecls.count(D))
          (*ParentStack.ReferencedDecls)[D] += DeclAndNumRefs.second;
      }
    }
    ScopeStack.pop_back();
  }
  return true;
}
