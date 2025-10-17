/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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

//===--- ConcreteDeclRef.cpp - Reference to a concrete decl ---------------===//
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
//
// This file implements the ConcreteDeclRef class, which provides a reference to
// a declaration that is potentially specialized.
//
//===----------------------------------------------------------------------===//

#include "language/AST/ASTContext.h"
#include "language/AST/ConcreteDeclRef.h"
#include "language/AST/Decl.h"
#include "language/AST/GenericSignature.h"
#include "language/AST/ProtocolConformance.h"
#include "language/AST/SubstitutionMap.h"
#include "language/AST/Types.h"
#include "language/Basic/Assertions.h"
#include "toolchain/Support/raw_ostream.h"
using namespace language;

ConcreteDeclRef ConcreteDeclRef::getOverriddenDecl() const {
  auto *derivedDecl = getDecl();
  auto *baseDecl = derivedDecl->getOverriddenDecl();

  auto baseSig = baseDecl->getInnermostDeclContext()
      ->getGenericSignatureOfContext();
  auto derivedSig = derivedDecl->getInnermostDeclContext()
      ->getGenericSignatureOfContext();

  SubstitutionMap subs;
  if (baseSig) {
    subs = SubstitutionMap::getOverrideSubstitutions(baseDecl, derivedDecl);
    if (derivedSig)
      subs = subs.subst(getSubstitutions());
  }
  return ConcreteDeclRef(baseDecl, subs);
}

ConcreteDeclRef ConcreteDeclRef::getOverriddenDecl(ValueDecl *baseDecl) const {
  auto *derivedDecl = getDecl();
  if (baseDecl == derivedDecl) return *this;

#ifndef NDEBUG
  {
    auto cur = derivedDecl;
    for (; cur && cur != baseDecl; cur = cur->getOverriddenDecl()) {}
    assert(cur && "decl is not an indirect override of baseDecl");
  }
#endif

  if (!baseDecl->getInnermostDeclContext()->isGenericContext()) {
    return ConcreteDeclRef(baseDecl);
  }

  auto subs = SubstitutionMap::getOverrideSubstitutions(baseDecl, derivedDecl);
  if (auto derivedSubs = getSubstitutions()) {
    subs = subs.subst(derivedSubs);
  }
  return ConcreteDeclRef(baseDecl, subs);
}

void ConcreteDeclRef::dump(raw_ostream &os) const {
  if (!getDecl()) {
    os << "**NULL**";
    return;
  }

  getDecl()->dumpRef(os);

  // If specialized, dump the substitutions.
  if (isSpecialized()) {
    os << " [with ";
    getSubstitutions().dump(os, SubstitutionMap::DumpStyle::Minimal);
    os << ']';
  }
}

void ConcreteDeclRef::dump() const {
  dump(toolchain::errs());
}
