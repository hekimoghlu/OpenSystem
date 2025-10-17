/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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

//===--- DerivedConformanceCaseIterable.cpp ---------------------*- C++ -*-===//
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
//  This file implements implicit derivation of the CaseIterable protocol.
//
//===----------------------------------------------------------------------===//

#include "DerivedConformance.h"
#include "TypeChecker.h"
#include "language/AST/Decl.h"
#include "language/AST/Expr.h"
#include "language/AST/Stmt.h"
#include "language/AST/Types.h"
#include "toolchain/Support/raw_ostream.h"

using namespace language;

/// Common preconditions for CaseIterable.
static bool canDeriveConformance(NominalTypeDecl *type) {
  // The type must be an enum.
  auto enumDecl = dyn_cast<EnumDecl>(type);
  if (!enumDecl)
    return false;

  // "Simple" enums without availability attributes can derive
  // a CaseIterable conformance.
  //
  // FIXME: Lift the availability restriction.
  return !enumDecl->hasPotentiallyUnavailableCaseValue()
      && enumDecl->hasOnlyCasesWithoutAssociatedValues();
}

/// Derive the implementation of allCases for a "simple" no-payload enum.
std::pair<BraceStmt *, bool>
deriveCaseIterable_enum_getter(AbstractFunctionDecl *funcDecl, void *) {
  auto *parentDC = funcDecl->getDeclContext();
  auto *parentEnum = parentDC->getSelfEnumDecl();
  auto enumTy = parentDC->getDeclaredTypeInContext();
  auto &C = parentDC->getASTContext();

  SmallVector<Expr *, 8> elExprs;
  for (EnumElementDecl *elt : parentEnum->getAllElements()) {
    auto *base = TypeExpr::createImplicit(enumTy, C);
    auto *apply = new (C) MemberRefExpr(base, SourceLoc(),
                                        elt, DeclNameLoc(), /*implicit*/true);
    elExprs.push_back(apply);
  }
  auto *arrayExpr = ArrayExpr::create(C, SourceLoc(), elExprs, {}, SourceLoc());

  auto *returnStmt = ReturnStmt::createImplicit(C, arrayExpr);
  auto *body = BraceStmt::create(C, SourceLoc(), ASTNode(returnStmt),
                                 SourceLoc());
  return { body, /*isTypeChecked=*/false };
}

static ArraySliceType *computeAllCasesType(NominalTypeDecl *enumDecl) {
  auto enumType = enumDecl->getDeclaredInterfaceType();
  if (!enumType || enumType->hasError())
    return nullptr;

  return ArraySliceType::get(enumType);
}

static Type deriveCaseIterable_AllCases(DerivedConformance &derived) {
  // enum SomeEnum : CaseIterable {
  //   @derived
  //   typealias AllCases = [SomeEnum]
  // }
  auto *rawInterfaceType = computeAllCasesType(cast<EnumDecl>(derived.Nominal));
  return derived.getConformanceContext()->mapTypeIntoContext(rawInterfaceType);
}

ValueDecl *DerivedConformance::deriveCaseIterable(ValueDecl *requirement) {
  auto &C = requirement->getASTContext();

  // Conformance can't be synthesized in an extension.
  if (checkAndDiagnoseDisallowedContext(requirement))
    return nullptr;

  // Check that we can actually derive CaseIterable for this type.
  if (!canDeriveConformance(Nominal))
    return nullptr;

  // Build the necessary decl.
  if (requirement->getBaseName() != Context.Id_allCases) {
    requirement->diagnose(diag::broken_case_iterable_requirement);
    return nullptr;
  }

  // Define the property.
  auto *returnTy = computeAllCasesType(Nominal);

  VarDecl *propDecl;
  PatternBindingDecl *pbDecl;
  std::tie(propDecl, pbDecl) = declareDerivedProperty(
      SynthesizedIntroducer::Var, Context.Id_allCases, returnTy,
      /*isStatic=*/true, /*isFinal=*/true);

  propDecl->getAttrs().add(NonisolatedAttr::createImplicit(C));

  // Define the getter.
  auto *getterDecl = addGetterToReadOnlyDerivedProperty(propDecl);

  getterDecl->setBodySynthesizer(&deriveCaseIterable_enum_getter);

  addMembersToConformanceContext({propDecl, pbDecl});

  return propDecl;
}

Type DerivedConformance::deriveCaseIterable(AssociatedTypeDecl *assocType) {
  // Check that we can actually derive CaseIterable for this type.
  if (!canDeriveConformance(Nominal))
    return nullptr;

  if (assocType->getName() == Context.Id_AllCases) {
    return deriveCaseIterable_AllCases(*this);
  }

  Context.Diags.diagnose(assocType->getLoc(),
                         diag::broken_case_iterable_requirement);
  return nullptr;
}

