/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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

//===--- NameLookup.cpp - Name lookup utilities ---------------------------===//
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

#include "NameLookup.h"
#include "language/AST/Decl.h"
#include "language/AST/GenericEnvironment.h"
#include "language/AST/Module.h"
#include "language/AST/Types.h"
#include "toolchain/ADT/SmallVector.h"
#include <algorithm>

using namespace language;
using namespace rewriting;

void
language::rewriting::lookupConcreteNestedType(
    Type baseType,
    Identifier name,
    SmallVectorImpl<TypeDecl *> &concreteDecls) {
  if (auto *decl = baseType->getAnyNominal())
    lookupConcreteNestedType(decl, name, concreteDecls);
  else if (auto *archetype = baseType->getAs<OpaqueTypeArchetypeType>()) {
    // If our concrete type is an opaque result archetype, look into its
    // generic environment recursively.
    auto *genericEnv = archetype->getGenericEnvironment();
    auto genericSig = genericEnv->getGenericSignature();

    auto *typeDecl =
        genericSig->lookupNestedType(archetype->getInterfaceType(), name);
    if (typeDecl != nullptr)
      concreteDecls.push_back(typeDecl);
  }
}

void
language::rewriting::lookupConcreteNestedType(
    NominalTypeDecl *decl,
    Identifier name,
    SmallVectorImpl<TypeDecl *> &concreteDecls) {
  SmallVector<ValueDecl *, 2> foundMembers;
  decl->getParentModule()->lookupQualified(
      decl, DeclNameRef(name), decl->getLoc(),
      NL_QualifiedDefault | NL_OnlyTypes | NL_ProtocolMembers,
      foundMembers);
  for (auto member : foundMembers)
    concreteDecls.push_back(cast<TypeDecl>(member));
}

TypeDecl *
language::rewriting::findBestConcreteNestedType(
    SmallVectorImpl<TypeDecl *> &concreteDecls) {
  if (concreteDecls.empty())
    return nullptr;

  return *std::min_element(concreteDecls.begin(), concreteDecls.end(),
                           [](TypeDecl *type1, TypeDecl *type2) {
                             return TypeDecl::compare(type1, type2) < 0;
                           });
}
