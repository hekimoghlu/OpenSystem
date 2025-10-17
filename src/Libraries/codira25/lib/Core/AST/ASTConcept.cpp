/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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

//===--- ASTConcept.cpp - Concepts Related AST Data Structures --*- C++ -*-===//
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
///
/// \file
/// \brief This file defines AST data structures related to concepts.
///
//===----------------------------------------------------------------------===//

#include "language/Core/AST/ASTConcept.h"
#include "language/Core/AST/ASTContext.h"
#include "language/Core/AST/ExprConcepts.h"
#include "language/Core/AST/NestedNameSpecifier.h"
#include "language/Core/AST/PrettyPrinter.h"
#include "toolchain/ADT/StringExtras.h"

using namespace language::Core;

static void
CreateUnsatisfiedConstraintRecord(const ASTContext &C,
                                  const UnsatisfiedConstraintRecord &Detail,
                                  UnsatisfiedConstraintRecord *TrailingObject) {
  if (auto *E = dyn_cast<Expr *>(Detail))
    new (TrailingObject) UnsatisfiedConstraintRecord(E);
  else {
    auto &SubstitutionDiagnostic =
        *cast<std::pair<SourceLocation, StringRef> *>(Detail);
    StringRef Message = C.backupStr(SubstitutionDiagnostic.second);
    auto *NewSubstDiag = new (C) std::pair<SourceLocation, StringRef>(
        SubstitutionDiagnostic.first, Message);
    new (TrailingObject) UnsatisfiedConstraintRecord(NewSubstDiag);
  }
}

ASTConstraintSatisfaction::ASTConstraintSatisfaction(
    const ASTContext &C, const ConstraintSatisfaction &Satisfaction)
    : NumRecords{Satisfaction.Details.size()},
      IsSatisfied{Satisfaction.IsSatisfied}, ContainsErrors{
                                                 Satisfaction.ContainsErrors} {
  for (unsigned I = 0; I < NumRecords; ++I)
    CreateUnsatisfiedConstraintRecord(C, Satisfaction.Details[I],
                                      getTrailingObjects() + I);
}

ASTConstraintSatisfaction::ASTConstraintSatisfaction(
    const ASTContext &C, const ASTConstraintSatisfaction &Satisfaction)
    : NumRecords{Satisfaction.NumRecords},
      IsSatisfied{Satisfaction.IsSatisfied},
      ContainsErrors{Satisfaction.ContainsErrors} {
  for (unsigned I = 0; I < NumRecords; ++I)
    CreateUnsatisfiedConstraintRecord(C, *(Satisfaction.begin() + I),
                                      getTrailingObjects() + I);
}

ASTConstraintSatisfaction *
ASTConstraintSatisfaction::Create(const ASTContext &C,
                                  const ConstraintSatisfaction &Satisfaction) {
  std::size_t size =
      totalSizeToAlloc<UnsatisfiedConstraintRecord>(
          Satisfaction.Details.size());
  void *Mem = C.Allocate(size, alignof(ASTConstraintSatisfaction));
  return new (Mem) ASTConstraintSatisfaction(C, Satisfaction);
}

ASTConstraintSatisfaction *ASTConstraintSatisfaction::Rebuild(
    const ASTContext &C, const ASTConstraintSatisfaction &Satisfaction) {
  std::size_t size =
      totalSizeToAlloc<UnsatisfiedConstraintRecord>(Satisfaction.NumRecords);
  void *Mem = C.Allocate(size, alignof(ASTConstraintSatisfaction));
  return new (Mem) ASTConstraintSatisfaction(C, Satisfaction);
}

void ConstraintSatisfaction::Profile(
    toolchain::FoldingSetNodeID &ID, const ASTContext &C,
    const NamedDecl *ConstraintOwner, ArrayRef<TemplateArgument> TemplateArgs) {
  ID.AddPointer(ConstraintOwner);
  ID.AddInteger(TemplateArgs.size());
  for (auto &Arg : TemplateArgs)
    Arg.Profile(ID, C);
}

ConceptReference *
ConceptReference::Create(const ASTContext &C, NestedNameSpecifierLoc NNS,
                         SourceLocation TemplateKWLoc,
                         DeclarationNameInfo ConceptNameInfo,
                         NamedDecl *FoundDecl, TemplateDecl *NamedConcept,
                         const ASTTemplateArgumentListInfo *ArgsAsWritten) {
  return new (C) ConceptReference(NNS, TemplateKWLoc, ConceptNameInfo,
                                  FoundDecl, NamedConcept, ArgsAsWritten);
}

SourceLocation ConceptReference::getBeginLoc() const {
  // Note that if the qualifier is null the template KW must also be null.
  if (auto QualifierLoc = getNestedNameSpecifierLoc())
    return QualifierLoc.getBeginLoc();
  return getConceptNameInfo().getBeginLoc();
}

void ConceptReference::print(toolchain::raw_ostream &OS,
                             const PrintingPolicy &Policy) const {
  NestedNameSpec.getNestedNameSpecifier().print(OS, Policy);
  ConceptName.printName(OS, Policy);
  if (hasExplicitTemplateArgs()) {
    OS << "<";
    toolchain::ListSeparator Sep(", ");
    // FIXME: Find corresponding parameter for argument
    for (auto &ArgLoc : ArgsAsWritten->arguments()) {
      OS << Sep;
      ArgLoc.getArgument().print(Policy, OS, /*IncludeType*/ false);
    }
    OS << ">";
  }
}

concepts::ExprRequirement::ExprRequirement(
    Expr *E, bool IsSimple, SourceLocation NoexceptLoc,
    ReturnTypeRequirement Req, SatisfactionStatus Status,
    ConceptSpecializationExpr *SubstitutedConstraintExpr)
    : Requirement(IsSimple ? RK_Simple : RK_Compound, Status == SS_Dependent,
                  Status == SS_Dependent &&
                      (E->containsUnexpandedParameterPack() ||
                       Req.containsUnexpandedParameterPack()),
                  Status == SS_Satisfied),
      Value(E), NoexceptLoc(NoexceptLoc), TypeReq(Req),
      SubstitutedConstraintExpr(SubstitutedConstraintExpr), Status(Status) {
  assert((!IsSimple || (Req.isEmpty() && NoexceptLoc.isInvalid())) &&
         "Simple requirement must not have a return type requirement or a "
         "noexcept specification");
  assert((Status > SS_TypeRequirementSubstitutionFailure &&
          Req.isTypeConstraint()) == (SubstitutedConstraintExpr != nullptr));
}

concepts::ExprRequirement::ExprRequirement(
    SubstitutionDiagnostic *ExprSubstDiag, bool IsSimple,
    SourceLocation NoexceptLoc, ReturnTypeRequirement Req)
    : Requirement(IsSimple ? RK_Simple : RK_Compound, Req.isDependent(),
                  Req.containsUnexpandedParameterPack(), /*IsSatisfied=*/false),
      Value(ExprSubstDiag), NoexceptLoc(NoexceptLoc), TypeReq(Req),
      Status(SS_ExprSubstitutionFailure) {
  assert((!IsSimple || (Req.isEmpty() && NoexceptLoc.isInvalid())) &&
         "Simple requirement must not have a return type requirement or a "
         "noexcept specification");
}

concepts::ExprRequirement::ReturnTypeRequirement::ReturnTypeRequirement(
    TemplateParameterList *TPL)
    : TypeConstraintInfo(TPL, false) {
  assert(TPL->size() == 1);
  const TypeConstraint *TC =
      cast<TemplateTypeParmDecl>(TPL->getParam(0))->getTypeConstraint();
  assert(TC &&
         "TPL must have a template type parameter with a type constraint");
  auto *Constraint =
      cast<ConceptSpecializationExpr>(TC->getImmediatelyDeclaredConstraint());
  bool Dependent =
      Constraint->getTemplateArgsAsWritten() &&
      TemplateSpecializationType::anyInstantiationDependentTemplateArguments(
          Constraint->getTemplateArgsAsWritten()->arguments().drop_front(1));
  TypeConstraintInfo.setInt(Dependent ? true : false);
}

concepts::ExprRequirement::ReturnTypeRequirement::ReturnTypeRequirement(
    TemplateParameterList *TPL, bool IsDependent)
    : TypeConstraintInfo(TPL, IsDependent) {}

concepts::TypeRequirement::TypeRequirement(TypeSourceInfo *T)
    : Requirement(RK_Type, T->getType()->isInstantiationDependentType(),
                  T->getType()->containsUnexpandedParameterPack(),
                  // We reach this ctor with either dependent types (in which
                  // IsSatisfied doesn't matter) or with non-dependent type in
                  // which the existence of the type indicates satisfaction.
                  /*IsSatisfied=*/true),
      Value(T),
      Status(T->getType()->isInstantiationDependentType() ? SS_Dependent
                                                          : SS_Satisfied) {}
