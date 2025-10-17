/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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

//=== PointerArithChecker.cpp - Pointer arithmetic checker -----*- C++ -*--===//
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
// This files defines PointerArithChecker, a builtin checker that checks for
// pointer arithmetic on locations other than array elements.
//
//===----------------------------------------------------------------------===//

#include "language/Core/AST/DeclCXX.h"
#include "language/Core/AST/ExprCXX.h"
#include "language/Core/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "language/Core/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "language/Core/StaticAnalyzer/Core/Checker.h"
#include "language/Core/StaticAnalyzer/Core/CheckerManager.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "toolchain/ADT/StringRef.h"

using namespace language::Core;
using namespace ento;

namespace {
enum class AllocKind {
  SingleObject,
  Array,
  Unknown,
  Reinterpreted // Single object interpreted as an array.
};
} // end namespace

namespace toolchain {
template <> struct FoldingSetTrait<AllocKind> {
  static inline void Profile(AllocKind X, FoldingSetNodeID &ID) {
    ID.AddInteger(static_cast<int>(X));
  }
};
} // end namespace toolchain

namespace {
class PointerArithChecker
    : public Checker<
          check::PreStmt<BinaryOperator>, check::PreStmt<UnaryOperator>,
          check::PreStmt<ArraySubscriptExpr>, check::PreStmt<CastExpr>,
          check::PostStmt<CastExpr>, check::PostStmt<CXXNewExpr>,
          check::PostStmt<CallExpr>, check::DeadSymbols> {
  AllocKind getKindOfNewOp(const CXXNewExpr *NE, const FunctionDecl *FD) const;
  const MemRegion *getArrayRegion(const MemRegion *Region, bool &Polymorphic,
                                  AllocKind &AKind, CheckerContext &C) const;
  const MemRegion *getPointedRegion(const MemRegion *Region,
                                    CheckerContext &C) const;
  void reportPointerArithMisuse(const Expr *E, CheckerContext &C,
                                bool PointedNeeded = false) const;
  void initAllocIdentifiers(ASTContext &C) const;

  const BugType BT_pointerArith{this, "Dangerous pointer arithmetic"};
  const BugType BT_polyArray{this, "Dangerous pointer arithmetic"};
  mutable toolchain::SmallSet<IdentifierInfo *, 8> AllocFunctions;

public:
  void checkPreStmt(const UnaryOperator *UOp, CheckerContext &C) const;
  void checkPreStmt(const BinaryOperator *BOp, CheckerContext &C) const;
  void checkPreStmt(const ArraySubscriptExpr *SubExpr, CheckerContext &C) const;
  void checkPreStmt(const CastExpr *CE, CheckerContext &C) const;
  void checkPostStmt(const CastExpr *CE, CheckerContext &C) const;
  void checkPostStmt(const CXXNewExpr *NE, CheckerContext &C) const;
  void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;
};
} // end namespace

REGISTER_MAP_WITH_PROGRAMSTATE(RegionState, const MemRegion *, AllocKind)

void PointerArithChecker::checkDeadSymbols(SymbolReaper &SR,
                                           CheckerContext &C) const {
  // TODO: intentional leak. Some information is garbage collected too early,
  // see http://reviews.toolchain.org/D14203 for further information.
  /*ProgramStateRef State = C.getState();
  RegionStateTy RegionStates = State->get<RegionState>();
  for (const MemRegion *Reg: toolchain::make_first_range(RegionStates)) {
    if (!SR.isLiveRegion(Reg))
      State = State->remove<RegionState>(Reg);
  }
  C.addTransition(State);*/
}

AllocKind PointerArithChecker::getKindOfNewOp(const CXXNewExpr *NE,
                                              const FunctionDecl *FD) const {
  // This checker try not to assume anything about placement and overloaded
  // new to avoid false positives.
  if (isa<CXXMethodDecl>(FD))
    return AllocKind::Unknown;
  if (FD->getNumParams() != 1 || FD->isVariadic())
    return AllocKind::Unknown;
  if (NE->isArray())
    return AllocKind::Array;

  return AllocKind::SingleObject;
}

const MemRegion *
PointerArithChecker::getPointedRegion(const MemRegion *Region,
                                      CheckerContext &C) const {
  assert(Region);
  ProgramStateRef State = C.getState();
  SVal S = State->getSVal(Region);
  return S.getAsRegion();
}

/// Checks whether a region is the part of an array.
/// In case there is a derived to base cast above the array element, the
/// Polymorphic output value is set to true. AKind output value is set to the
/// allocation kind of the inspected region.
const MemRegion *PointerArithChecker::getArrayRegion(const MemRegion *Region,
                                                     bool &Polymorphic,
                                                     AllocKind &AKind,
                                                     CheckerContext &C) const {
  assert(Region);
  while (const auto *BaseRegion = dyn_cast<CXXBaseObjectRegion>(Region)) {
    Region = BaseRegion->getSuperRegion();
    Polymorphic = true;
  }
  if (const auto *ElemRegion = dyn_cast<ElementRegion>(Region)) {
    Region = ElemRegion->getSuperRegion();
  }

  ProgramStateRef State = C.getState();
  if (const AllocKind *Kind = State->get<RegionState>(Region)) {
    AKind = *Kind;
    if (*Kind == AllocKind::Array)
      return Region;
    else
      return nullptr;
  }
  // When the region is symbolic and we do not have any information about it,
  // assume that this is an array to avoid false positives.
  if (isa<SymbolicRegion>(Region))
    return Region;

  // No AllocKind stored and not symbolic, assume that it points to a single
  // object.
  return nullptr;
}

void PointerArithChecker::reportPointerArithMisuse(const Expr *E,
                                                   CheckerContext &C,
                                                   bool PointedNeeded) const {
  SourceRange SR = E->getSourceRange();
  if (SR.isInvalid())
    return;

  ProgramStateRef State = C.getState();
  const MemRegion *Region = C.getSVal(E).getAsRegion();
  if (!Region)
    return;
  if (PointedNeeded)
    Region = getPointedRegion(Region, C);
  if (!Region)
    return;

  bool IsPolymorphic = false;
  AllocKind Kind = AllocKind::Unknown;
  if (const MemRegion *ArrayRegion =
          getArrayRegion(Region, IsPolymorphic, Kind, C)) {
    if (!IsPolymorphic)
      return;
    if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
      constexpr toolchain::StringLiteral Msg =
          "Pointer arithmetic on a pointer to base class is dangerous "
          "because derived and base class may have different size.";
      auto R = std::make_unique<PathSensitiveBugReport>(BT_polyArray, Msg, N);
      R->addRange(E->getSourceRange());
      R->markInteresting(ArrayRegion);
      C.emitReport(std::move(R));
    }
    return;
  }

  if (Kind == AllocKind::Reinterpreted)
    return;

  // We might not have enough information about symbolic regions.
  if (Kind != AllocKind::SingleObject &&
      Region->getKind() == MemRegion::Kind::SymbolicRegionKind)
    return;

  if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
    constexpr toolchain::StringLiteral Msg =
        "Pointer arithmetic on non-array variables relies on memory layout, "
        "which is dangerous.";
    auto R = std::make_unique<PathSensitiveBugReport>(BT_pointerArith, Msg, N);
    R->addRange(SR);
    R->markInteresting(Region);
    C.emitReport(std::move(R));
  }
}

void PointerArithChecker::initAllocIdentifiers(ASTContext &C) const {
  if (!AllocFunctions.empty())
    return;
  AllocFunctions.insert(&C.Idents.get("alloca"));
  AllocFunctions.insert(&C.Idents.get("malloc"));
  AllocFunctions.insert(&C.Idents.get("realloc"));
  AllocFunctions.insert(&C.Idents.get("calloc"));
  AllocFunctions.insert(&C.Idents.get("valloc"));
}

void PointerArithChecker::checkPostStmt(const CallExpr *CE,
                                        CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const FunctionDecl *FD = C.getCalleeDecl(CE);
  if (!FD)
    return;
  IdentifierInfo *FunI = FD->getIdentifier();
  initAllocIdentifiers(C.getASTContext());
  if (AllocFunctions.count(FunI) == 0)
    return;

  SVal SV = C.getSVal(CE);
  const MemRegion *Region = SV.getAsRegion();
  if (!Region)
    return;
  // Assume that C allocation functions allocate arrays to avoid false
  // positives.
  // TODO: Add heuristics to distinguish alloc calls that allocates single
  // objecs.
  State = State->set<RegionState>(Region, AllocKind::Array);
  C.addTransition(State);
}

void PointerArithChecker::checkPostStmt(const CXXNewExpr *NE,
                                        CheckerContext &C) const {
  const FunctionDecl *FD = NE->getOperatorNew();
  if (!FD)
    return;

  AllocKind Kind = getKindOfNewOp(NE, FD);

  ProgramStateRef State = C.getState();
  SVal AllocedVal = C.getSVal(NE);
  const MemRegion *Region = AllocedVal.getAsRegion();
  if (!Region)
    return;
  State = State->set<RegionState>(Region, Kind);
  C.addTransition(State);
}

void PointerArithChecker::checkPostStmt(const CastExpr *CE,
                                        CheckerContext &C) const {
  if (CE->getCastKind() != CastKind::CK_BitCast)
    return;

  const Expr *CastedExpr = CE->getSubExpr();
  ProgramStateRef State = C.getState();
  SVal CastedVal = C.getSVal(CastedExpr);

  const MemRegion *Region = CastedVal.getAsRegion();
  if (!Region)
    return;

  // Suppress reinterpret casted hits.
  State = State->set<RegionState>(Region, AllocKind::Reinterpreted);
  C.addTransition(State);
}

void PointerArithChecker::checkPreStmt(const CastExpr *CE,
                                       CheckerContext &C) const {
  if (CE->getCastKind() != CastKind::CK_ArrayToPointerDecay)
    return;

  const Expr *CastedExpr = CE->getSubExpr();
  ProgramStateRef State = C.getState();
  SVal CastedVal = C.getSVal(CastedExpr);

  const MemRegion *Region = CastedVal.getAsRegion();
  if (!Region)
    return;

  if (const AllocKind *Kind = State->get<RegionState>(Region)) {
    if (*Kind == AllocKind::Array || *Kind == AllocKind::Reinterpreted)
      return;
  }
  State = State->set<RegionState>(Region, AllocKind::Array);
  C.addTransition(State);
}

void PointerArithChecker::checkPreStmt(const UnaryOperator *UOp,
                                       CheckerContext &C) const {
  if (!UOp->isIncrementDecrementOp() || !UOp->getType()->isPointerType())
    return;
  reportPointerArithMisuse(UOp->getSubExpr(), C, true);
}

void PointerArithChecker::checkPreStmt(const ArraySubscriptExpr *SubsExpr,
                                       CheckerContext &C) const {
  SVal Idx = C.getSVal(SubsExpr->getIdx());

  // Indexing with 0 is OK.
  if (Idx.isZeroConstant())
    return;

  // Indexing vector-type expressions is also OK.
  if (SubsExpr->getBase()->getType()->isVectorType())
    return;
  reportPointerArithMisuse(SubsExpr->getBase(), C);
}

void PointerArithChecker::checkPreStmt(const BinaryOperator *BOp,
                                       CheckerContext &C) const {
  BinaryOperatorKind OpKind = BOp->getOpcode();
  if (!BOp->isAdditiveOp() && OpKind != BO_AddAssign && OpKind != BO_SubAssign)
    return;

  const Expr *Lhs = BOp->getLHS();
  const Expr *Rhs = BOp->getRHS();
  ProgramStateRef State = C.getState();

  if (Rhs->getType()->isIntegerType() && Lhs->getType()->isPointerType()) {
    SVal RHSVal = C.getSVal(Rhs);
    if (State->isNull(RHSVal).isConstrainedTrue())
      return;
    reportPointerArithMisuse(Lhs, C, !BOp->isAdditiveOp());
  }
  // The int += ptr; case is not valid C++.
  if (Lhs->getType()->isIntegerType() && Rhs->getType()->isPointerType()) {
    SVal LHSVal = C.getSVal(Lhs);
    if (State->isNull(LHSVal).isConstrainedTrue())
      return;
    reportPointerArithMisuse(Rhs, C);
  }
}

void ento::registerPointerArithChecker(CheckerManager &mgr) {
  mgr.registerChecker<PointerArithChecker>();
}

bool ento::shouldRegisterPointerArithChecker(const CheckerManager &mgr) {
  return true;
}
