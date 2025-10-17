/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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
#include "language/Basic/Assertions.h"

using namespace language::refactoring;

bool RefactoringActionMoveMembersToExtension::isApplicable(
    const ResolvedRangeInfo &Info, DiagnosticEngine &Diag) {
  switch (Info.Kind) {
  case RangeKind::SingleDecl:
  case RangeKind::MultiTypeMemberDecl: {
    DeclContext *DC = Info.RangeContext;

    // The common decl context is not a nomial type, we cannot create an
    // extension for it
    if (!DC || !DC->getInnermostDeclarationDeclContext() ||
        !isa<NominalTypeDecl>(DC->getInnermostDeclarationDeclContext()))
      return false;

    // Members of types not declared at top file level cannot be extracted
    // to an extension at top file level
    if (DC->getParent()->getContextKind() != DeclContextKind::FileUnit)
      return false;

    // Check if contained nodes are all allowed decls.
    for (auto Node : Info.ContainedNodes) {
      Decl *D = Node.dyn_cast<Decl *>();
      if (!D)
        return false;

      if (isa<AccessorDecl>(D) || isa<DestructorDecl>(D) ||
          isa<EnumCaseDecl>(D) || isa<EnumElementDecl>(D))
        return false;
    }

    // We should not move instance variables with storage into the extension
    // because they are not allowed to be declared there
    for (auto DD : Info.DeclaredDecls) {
      if (auto ASD = dyn_cast<AbstractStorageDecl>(DD.VD)) {
        // Only disallow storages in the common decl context, allow them in
        // any subtypes
        if (ASD->hasStorage() && ASD->getDeclContext() == DC) {
          return false;
        }
      }
    }

    return true;
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

bool RefactoringActionMoveMembersToExtension::performChange() {
  DeclContext *DC = RangeInfo.RangeContext;

  auto CommonTypeDecl =
      dyn_cast<NominalTypeDecl>(DC->getInnermostDeclarationDeclContext());
  assert(CommonTypeDecl && "Not applicable if common parent is no nomial type");

  SmallString<64> Buffer;
  toolchain::raw_svector_ostream OS(Buffer);
  OS << "\n\n";
  OS << "extension " << CommonTypeDecl->getName() << " {\n";
  OS << RangeInfo.ContentRange.str().trim();
  OS << "\n}";

  // Insert extension after the type declaration
  EditConsumer.insertAfter(SM, CommonTypeDecl->getEndLoc(), Buffer);
  EditConsumer.remove(SM, RangeInfo.ContentRange);

  return false;
}
