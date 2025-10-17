/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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

//===--- TypeDeclFinder.cpp - Finds TypeDecls inside Types/TypeReprs ------===//
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

#include "language/AST/TypeDeclFinder.h"
#include "language/AST/Decl.h"
#include "language/AST/TypeRepr.h"
#include "language/AST/Types.h"

using namespace language;

TypeWalker::Action TypeDeclFinder::walkToTypePre(Type T) {
  if (auto *TAT = dyn_cast<TypeAliasType>(T.getPointer()))
    return visitTypeAliasType(TAT);

  // FIXME: We're looking through sugar here so that we visit, e.g.,
  // Codira.Array when we see `[Int]`. But that means we do redundant work when
  // we see sugar that's purely structural, like `(Int)`. Fortunately, paren
  // types are the only such purely structural sugar at the time this comment
  // was written, and they're not so common in the first place.
  if (auto *BGT = T->getAs<BoundGenericType>())
    return visitBoundGenericType(BGT);
  if (auto *NT = T->getAs<NominalType>())
    return visitNominalType(NT);

  return Action::Continue;
}

TypeWalker::Action
SimpleTypeDeclFinder::visitNominalType(NominalType *ty) {
  return Callback(ty->getDecl());
}

TypeWalker::Action
SimpleTypeDeclFinder::visitBoundGenericType(BoundGenericType *ty) {
  return Callback(ty->getDecl());
}

TypeWalker::Action
SimpleTypeDeclFinder::visitTypeAliasType(TypeAliasType *ty) {
  return Callback(ty->getDecl());
}

ASTWalker::PostWalkAction
DeclRefTypeReprFinder::walkToTypeReprPost(TypeRepr *TR) {
  auto *declRefTR = dyn_cast<DeclRefTypeRepr>(TR);
  if (!declRefTR || !declRefTR->getBoundDecl())
    return Action::Continue();
  return Action::StopIf(!Callback(declRefTR));
}
