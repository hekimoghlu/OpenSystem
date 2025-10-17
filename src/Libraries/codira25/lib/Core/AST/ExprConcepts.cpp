/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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

//===- ExprCXX.cpp - (C++) Expression AST Node Implementation -------------===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Expr class declared in ExprCXX.h
//
//===----------------------------------------------------------------------===//

#include "language/Core/AST/ExprConcepts.h"
#include "language/Core/AST/ASTConcept.h"
#include "language/Core/AST/ASTContext.h"
#include "language/Core/AST/ComputeDependence.h"
#include "language/Core/AST/Decl.h"
#include "language/Core/AST/DeclTemplate.h"
#include "language/Core/AST/DeclarationName.h"
#include "language/Core/AST/DependenceFlags.h"
#include "language/Core/AST/Expr.h"
#include "language/Core/AST/NestedNameSpecifier.h"
#include "language/Core/AST/TemplateBase.h"
#include "language/Core/AST/Type.h"
#include "language/Core/Basic/SourceLocation.h"

using namespace language::Core;

ConceptSpecializationExpr::ConceptSpecializationExpr(
    const ASTContext &C, ConceptReference *Loc,
    ImplicitConceptSpecializationDecl *SpecDecl,
    const ConstraintSatisfaction *Satisfaction)
    : Expr(ConceptSpecializationExprClass, C.BoolTy, VK_PRValue, OK_Ordinary),
      ConceptRef(Loc), SpecDecl(SpecDecl),
      Satisfaction(Satisfaction
                       ? ASTConstraintSatisfaction::Create(C, *Satisfaction)
                       : nullptr) {
  setDependence(computeDependence(this, /*ValueDependent=*/!Satisfaction));

  // Currently guaranteed by the fact concepts can only be at namespace-scope.
  assert(!Loc->getNestedNameSpecifierLoc() ||
         (!Loc->getNestedNameSpecifierLoc()
               .getNestedNameSpecifier()
               .isInstantiationDependent() &&
          !Loc->getNestedNameSpecifierLoc()
               .getNestedNameSpecifier()
               .containsUnexpandedParameterPack()));
  assert((!isValueDependent() || isInstantiationDependent()) &&
         "should not be value-dependent");
}

ConceptSpecializationExpr::ConceptSpecializationExpr(EmptyShell Empty)
    : Expr(ConceptSpecializationExprClass, Empty) {}

ConceptSpecializationExpr *
ConceptSpecializationExpr::Create(const ASTContext &C, ConceptReference *Loc,
                                  ImplicitConceptSpecializationDecl *SpecDecl,
                                  const ConstraintSatisfaction *Satisfaction) {
  return new (C) ConceptSpecializationExpr(C, Loc, SpecDecl, Satisfaction);
}

ConceptSpecializationExpr::ConceptSpecializationExpr(
    const ASTContext &C, ConceptReference *Loc,
    ImplicitConceptSpecializationDecl *SpecDecl,
    const ConstraintSatisfaction *Satisfaction, bool Dependent,
    bool ContainsUnexpandedParameterPack)
    : Expr(ConceptSpecializationExprClass, C.BoolTy, VK_PRValue, OK_Ordinary),
      ConceptRef(Loc), SpecDecl(SpecDecl),
      Satisfaction(Satisfaction
                       ? ASTConstraintSatisfaction::Create(C, *Satisfaction)
                       : nullptr) {
  ExprDependence D = ExprDependence::None;
  if (!Satisfaction)
    D |= ExprDependence::Value;
  if (Dependent)
    D |= ExprDependence::Instantiation;
  if (ContainsUnexpandedParameterPack)
    D |= ExprDependence::UnexpandedPack;
  setDependence(D);
}

ConceptSpecializationExpr *
ConceptSpecializationExpr::Create(const ASTContext &C, ConceptReference *Loc,
                                  ImplicitConceptSpecializationDecl *SpecDecl,
                                  const ConstraintSatisfaction *Satisfaction,
                                  bool Dependent,
                                  bool ContainsUnexpandedParameterPack) {
  return new (C)
      ConceptSpecializationExpr(C, Loc, SpecDecl, Satisfaction, Dependent,
                                ContainsUnexpandedParameterPack);
}

const TypeConstraint *
concepts::ExprRequirement::ReturnTypeRequirement::getTypeConstraint() const {
  assert(isTypeConstraint());
  auto TPL = cast<TemplateParameterList *>(TypeConstraintInfo.getPointer());
  return cast<TemplateTypeParmDecl>(TPL->getParam(0))
      ->getTypeConstraint();
}

// Search through the requirements, and see if any have a RecoveryExpr in it,
// which means this RequiresExpr ALSO needs to be invalid.
static bool RequirementContainsError(concepts::Requirement *R) {
  if (auto *ExprReq = dyn_cast<concepts::ExprRequirement>(R))
    return ExprReq->getExpr() && ExprReq->getExpr()->containsErrors();

  if (auto *NestedReq = dyn_cast<concepts::NestedRequirement>(R))
    return !NestedReq->hasInvalidConstraint() &&
           NestedReq->getConstraintExpr() &&
           NestedReq->getConstraintExpr()->containsErrors();
  return false;
}

RequiresExpr::RequiresExpr(ASTContext &C, SourceLocation RequiresKWLoc,
                           RequiresExprBodyDecl *Body, SourceLocation LParenLoc,
                           ArrayRef<ParmVarDecl *> LocalParameters,
                           SourceLocation RParenLoc,
                           ArrayRef<concepts::Requirement *> Requirements,
                           SourceLocation RBraceLoc)
    : Expr(RequiresExprClass, C.BoolTy, VK_PRValue, OK_Ordinary),
      NumLocalParameters(LocalParameters.size()),
      NumRequirements(Requirements.size()), Body(Body), LParenLoc(LParenLoc),
      RParenLoc(RParenLoc), RBraceLoc(RBraceLoc) {
  RequiresExprBits.IsSatisfied = false;
  RequiresExprBits.RequiresKWLoc = RequiresKWLoc;
  bool Dependent = false;
  bool ContainsUnexpandedParameterPack = false;
  for (ParmVarDecl *P : LocalParameters) {
    Dependent |= P->getType()->isInstantiationDependentType();
    ContainsUnexpandedParameterPack |=
        P->getType()->containsUnexpandedParameterPack();
  }
  RequiresExprBits.IsSatisfied = true;
  for (concepts::Requirement *R : Requirements) {
    Dependent |= R->isDependent();
    ContainsUnexpandedParameterPack |= R->containsUnexpandedParameterPack();
    if (!Dependent) {
      RequiresExprBits.IsSatisfied = R->isSatisfied();
      if (!RequiresExprBits.IsSatisfied)
        break;
    }

    if (RequirementContainsError(R))
      setDependence(getDependence() | ExprDependence::Error);
  }
  toolchain::copy(LocalParameters, getTrailingObjects<ParmVarDecl *>());
  toolchain::copy(Requirements, getTrailingObjects<concepts::Requirement *>());
  RequiresExprBits.IsSatisfied |= Dependent;
  // FIXME: move the computing dependency logic to ComputeDependence.h
  if (ContainsUnexpandedParameterPack)
    setDependence(getDependence() | ExprDependence::UnexpandedPack);
  // FIXME: this is incorrect for cases where we have a non-dependent
  // requirement, but its parameters are instantiation-dependent. RequiresExpr
  // should be instantiation-dependent if it has instantiation-dependent
  // parameters.
  if (Dependent)
    setDependence(getDependence() | ExprDependence::ValueInstantiation);
}

RequiresExpr::RequiresExpr(ASTContext &C, EmptyShell Empty,
                           unsigned NumLocalParameters,
                           unsigned NumRequirements)
  : Expr(RequiresExprClass, Empty), NumLocalParameters(NumLocalParameters),
    NumRequirements(NumRequirements) { }

RequiresExpr *RequiresExpr::Create(
    ASTContext &C, SourceLocation RequiresKWLoc, RequiresExprBodyDecl *Body,
    SourceLocation LParenLoc, ArrayRef<ParmVarDecl *> LocalParameters,
    SourceLocation RParenLoc, ArrayRef<concepts::Requirement *> Requirements,
    SourceLocation RBraceLoc) {
  void *Mem =
      C.Allocate(totalSizeToAlloc<ParmVarDecl *, concepts::Requirement *>(
                     LocalParameters.size(), Requirements.size()),
                 alignof(RequiresExpr));
  return new (Mem)
      RequiresExpr(C, RequiresKWLoc, Body, LParenLoc, LocalParameters,
                   RParenLoc, Requirements, RBraceLoc);
}

RequiresExpr *
RequiresExpr::Create(ASTContext &C, EmptyShell Empty,
                     unsigned NumLocalParameters, unsigned NumRequirements) {
  void *Mem =
      C.Allocate(totalSizeToAlloc<ParmVarDecl *, concepts::Requirement *>(
                     NumLocalParameters, NumRequirements),
                 alignof(RequiresExpr));
  return new (Mem) RequiresExpr(C, Empty, NumLocalParameters, NumRequirements);
}
