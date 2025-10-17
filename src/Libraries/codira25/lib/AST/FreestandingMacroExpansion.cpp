/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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

//===--- FreestandingMacroExpansion.cpp -----------------------------------===//
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

#include "language/AST/FreestandingMacroExpansion.h"
#include "language/AST/ASTContext.h"
#include "language/AST/Decl.h"
#include "language/AST/Expr.h"
#include "language/AST/MacroDiscriminatorContext.h"
#include "language/Basic/Assertions.h"

using namespace language;

SourceRange MacroExpansionInfo::getSourceRange() const {
  SourceLoc endLoc;
  if (ArgList && !ArgList->isImplicit())
    endLoc = ArgList->getEndLoc();
  else if (RightAngleLoc.isValid())
    endLoc = RightAngleLoc;
  else
    endLoc = MacroNameLoc.getEndLoc();

  return SourceRange(SigilLoc, endLoc);
}

#define FORWARD_VARIANT(NAME)                                                  \
  switch (getFreestandingMacroKind()) {                                        \
  case FreestandingMacroKind::Expr:                                            \
    return cast<MacroExpansionExpr>(this)->NAME();                             \
  case FreestandingMacroKind::Decl:                                            \
    return cast<MacroExpansionDecl>(this)->NAME();                             \
  }

DeclContext *FreestandingMacroExpansion::getDeclContext() const {
  FORWARD_VARIANT(getDeclContext);
}
SourceRange FreestandingMacroExpansion::getSourceRange() const {
  FORWARD_VARIANT(getSourceRange);
}
unsigned FreestandingMacroExpansion::getDiscriminator() const {
  auto info = getExpansionInfo();
  if (info->Discriminator != MacroExpansionInfo::InvalidDiscriminator)
    return info->Discriminator;

  auto mutableThis = const_cast<FreestandingMacroExpansion *>(this);
  auto dc = getDeclContext();
  ASTContext &ctx = dc->getASTContext();
  auto discriminatorContext =
      MacroDiscriminatorContext::getParentOf(mutableThis);
  info->Discriminator = ctx.getNextMacroDiscriminator(
      discriminatorContext, getMacroName().getBaseName());

  assert(info->Discriminator != MacroExpansionInfo::InvalidDiscriminator);
  return info->Discriminator;
}

unsigned FreestandingMacroExpansion::getRawDiscriminator() const {
  return getExpansionInfo()->Discriminator;
}

ASTNode FreestandingMacroExpansion::getASTNode() {
  switch (getFreestandingMacroKind()) {
  case FreestandingMacroKind::Expr:
    return cast<MacroExpansionExpr>(this);
  case FreestandingMacroKind::Decl:
    return cast<MacroExpansionDecl>(this);
  }
}
