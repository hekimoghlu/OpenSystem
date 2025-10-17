/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#include "Utils.h"

using namespace language;
using namespace language::refactoring::asyncrefactorings;

void DeclCollector::collect(BraceStmt *Scope, SourceFile &SF,
                            toolchain::DenseSet<const Decl *> &Decls) {
  DeclCollector Collector(Decls);
  if (Scope) {
    for (auto Node : Scope->getElements()) {
      Collector.walk(Node);
    }
  } else {
    Collector.walk(SF);
  }
}

bool DeclCollector::walkToDeclPre(Decl *D, CharSourceRange Range) {
  // Want to walk through top level code decls (which are implicitly added
  // for top level non-decl code) and pattern binding decls (which contain
  // the var decls that we care about).
  if (isa<TopLevelCodeDecl>(D) || isa<PatternBindingDecl>(D))
    return true;

  if (!D->isImplicit())
    Decls.insert(D);
  return false;
}

bool DeclCollector::walkToExprPre(Expr *E) { return !isa<ClosureExpr>(E); }

bool DeclCollector::walkToStmtPre(Stmt *S) {
  return S->isImplicit() || !startsNewScope(S);
}
