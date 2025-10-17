/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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

//== DynamicTypeChecker.cpp ------------------------------------ -*- C++ -*--=//
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
// This checker looks for cases where the dynamic type of an object is unrelated
// to its static type. The type information utilized by this check is collected
// by the DynamicTypePropagation checker. This check does not report any type
// error for ObjC Generic types, in order to avoid duplicate erros from the
// ObjC Generics checker. This checker is not supposed to modify the program
// state, it is just the observer of the type information provided by other
// checkers.
//
//===----------------------------------------------------------------------===//

#include "language/Core/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "language/Core/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "language/Core/StaticAnalyzer/Core/Checker.h"
#include "language/Core/StaticAnalyzer/Core/CheckerManager.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/DynamicType.h"

using namespace language::Core;
using namespace ento;

namespace {
class DynamicTypeChecker : public Checker<check::PostStmt<ImplicitCastExpr>> {
  const BugType BT{this, "Dynamic and static type mismatch", "Type Error"};

  class DynamicTypeBugVisitor : public BugReporterVisitor {
  public:
    DynamicTypeBugVisitor(const MemRegion *Reg) : Reg(Reg) {}

    void Profile(toolchain::FoldingSetNodeID &ID) const override {
      static int X = 0;
      ID.AddPointer(&X);
      ID.AddPointer(Reg);
    }

    PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                     BugReporterContext &BRC,
                                     PathSensitiveBugReport &BR) override;

  private:
    // The tracked region.
    const MemRegion *Reg;
  };

  void reportTypeError(QualType DynamicType, QualType StaticType,
                       const MemRegion *Reg, const Stmt *ReportedNode,
                       CheckerContext &C) const;

public:
  void checkPostStmt(const ImplicitCastExpr *CE, CheckerContext &C) const;
};
}

void DynamicTypeChecker::reportTypeError(QualType DynamicType,
                                         QualType StaticType,
                                         const MemRegion *Reg,
                                         const Stmt *ReportedNode,
                                         CheckerContext &C) const {
  SmallString<192> Buf;
  toolchain::raw_svector_ostream OS(Buf);
  OS << "Object has a dynamic type '";
  QualType::print(DynamicType.getTypePtr(), Qualifiers(), OS, C.getLangOpts(),
                  toolchain::Twine());
  OS << "' which is incompatible with static type '";
  QualType::print(StaticType.getTypePtr(), Qualifiers(), OS, C.getLangOpts(),
                  toolchain::Twine());
  OS << "'";
  auto R = std::make_unique<PathSensitiveBugReport>(
      BT, OS.str(), C.generateNonFatalErrorNode());
  R->markInteresting(Reg);
  R->addVisitor(std::make_unique<DynamicTypeBugVisitor>(Reg));
  R->addRange(ReportedNode->getSourceRange());
  C.emitReport(std::move(R));
}

PathDiagnosticPieceRef DynamicTypeChecker::DynamicTypeBugVisitor::VisitNode(
    const ExplodedNode *N, BugReporterContext &BRC, PathSensitiveBugReport &) {
  ProgramStateRef State = N->getState();
  ProgramStateRef StatePrev = N->getFirstPred()->getState();

  DynamicTypeInfo TrackedType = getDynamicTypeInfo(State, Reg);
  DynamicTypeInfo TrackedTypePrev = getDynamicTypeInfo(StatePrev, Reg);
  if (!TrackedType.isValid())
    return nullptr;

  if (TrackedTypePrev.isValid() &&
      TrackedTypePrev.getType() == TrackedType.getType())
    return nullptr;

  // Retrieve the associated statement.
  const Stmt *S = N->getStmtForDiagnostics();
  if (!S)
    return nullptr;

  const LangOptions &LangOpts = BRC.getASTContext().getLangOpts();

  SmallString<256> Buf;
  toolchain::raw_svector_ostream OS(Buf);
  OS << "Type '";
  QualType::print(TrackedType.getType().getTypePtr(), Qualifiers(), OS,
                  LangOpts, toolchain::Twine());
  OS << "' is inferred from ";

  if (const auto *ExplicitCast = dyn_cast<ExplicitCastExpr>(S)) {
    OS << "explicit cast (from '";
    QualType::print(ExplicitCast->getSubExpr()->getType().getTypePtr(),
                    Qualifiers(), OS, LangOpts, toolchain::Twine());
    OS << "' to '";
    QualType::print(ExplicitCast->getType().getTypePtr(), Qualifiers(), OS,
                    LangOpts, toolchain::Twine());
    OS << "')";
  } else if (const auto *ImplicitCast = dyn_cast<ImplicitCastExpr>(S)) {
    OS << "implicit cast (from '";
    QualType::print(ImplicitCast->getSubExpr()->getType().getTypePtr(),
                    Qualifiers(), OS, LangOpts, toolchain::Twine());
    OS << "' to '";
    QualType::print(ImplicitCast->getType().getTypePtr(), Qualifiers(), OS,
                    LangOpts, toolchain::Twine());
    OS << "')";
  } else {
    OS << "this context";
  }

  // Generate the extra diagnostic.
  PathDiagnosticLocation Pos(S, BRC.getSourceManager(),
                             N->getLocationContext());
  return std::make_shared<PathDiagnosticEventPiece>(Pos, OS.str(), true);
}

static bool hasDefinition(const ObjCObjectPointerType *ObjPtr) {
  const ObjCInterfaceDecl *Decl = ObjPtr->getInterfaceDecl();
  if (!Decl)
    return false;

  return Decl->getDefinition();
}

// TODO: consider checking explicit casts?
void DynamicTypeChecker::checkPostStmt(const ImplicitCastExpr *CE,
                                       CheckerContext &C) const {
  // TODO: C++ support.
  if (CE->getCastKind() != CK_BitCast)
    return;

  const MemRegion *Region = C.getSVal(CE).getAsRegion();
  if (!Region)
    return;

  ProgramStateRef State = C.getState();
  DynamicTypeInfo DynTypeInfo = getDynamicTypeInfo(State, Region);

  if (!DynTypeInfo.isValid())
    return;

  QualType DynType = DynTypeInfo.getType();
  QualType StaticType = CE->getType();

  const auto *DynObjCType = DynType->getAs<ObjCObjectPointerType>();
  const auto *StaticObjCType = StaticType->getAs<ObjCObjectPointerType>();

  if (!DynObjCType || !StaticObjCType)
    return;

  if (!hasDefinition(DynObjCType) || !hasDefinition(StaticObjCType))
    return;

  ASTContext &ASTCtxt = C.getASTContext();

  // Strip kindeofness to correctly detect subtyping relationships.
  DynObjCType = DynObjCType->stripObjCKindOfTypeAndQuals(ASTCtxt);
  StaticObjCType = StaticObjCType->stripObjCKindOfTypeAndQuals(ASTCtxt);

  // Specialized objects are handled by the generics checker.
  if (StaticObjCType->isSpecialized())
    return;

  if (ASTCtxt.canAssignObjCInterfaces(StaticObjCType, DynObjCType))
    return;

  if (DynTypeInfo.canBeASubClass() &&
      ASTCtxt.canAssignObjCInterfaces(DynObjCType, StaticObjCType))
    return;

  reportTypeError(DynType, StaticType, Region, CE, C);
}

void ento::registerDynamicTypeChecker(CheckerManager &mgr) {
  mgr.registerChecker<DynamicTypeChecker>();
}

bool ento::shouldRegisterDynamicTypeChecker(const CheckerManager &mgr) {
  return true;
}
