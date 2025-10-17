/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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

//===- GCDAntipatternChecker.cpp ---------------------------------*- C++ -*-==//
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
// This file defines GCDAntipatternChecker which checks against a common
// antipattern when synchronous API is emulated from asynchronous callbacks
// using a semaphore:
//
//   dispatch_semaphore_t sema = dispatch_semaphore_create(0);
//
//   AnyCFunctionCall(^{
//     // codeâ€¦
//     dispatch_semaphore_signal(sema);
//   })
//   dispatch_semaphore_wait(sema, *)
//
// Such code is a common performance problem, due to inability of GCD to
// properly handle QoS when a combination of queues and semaphores is used.
// Good code would either use asynchronous API (when available), or perform
// the necessary action in asynchronous callback.
//
// Currently, the check is performed using a simple heuristical AST pattern
// matching.
//
//===----------------------------------------------------------------------===//

#include "language/Core/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "language/Core/ASTMatchers/ASTMatchFinder.h"
#include "language/Core/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "language/Core/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "language/Core/StaticAnalyzer/Core/Checker.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "toolchain/Support/Debug.h"

using namespace language::Core;
using namespace ento;
using namespace ast_matchers;

namespace {

// ID of a node at which the diagnostic would be emitted.
const char *WarnAtNode = "waitcall";

class GCDAntipatternChecker : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D,
                        AnalysisManager &AM,
                        BugReporter &BR) const;
};

decltype(auto) callsName(const char *FunctionName) {
  return callee(functionDecl(hasName(FunctionName)));
}

decltype(auto) equalsBoundArgDecl(int ArgIdx, const char *DeclName) {
  return hasArgument(ArgIdx, ignoringParenCasts(declRefExpr(
                                 to(varDecl(equalsBoundNode(DeclName))))));
}

decltype(auto) bindAssignmentToDecl(const char *DeclName) {
  return hasLHS(ignoringParenImpCasts(
                         declRefExpr(to(varDecl().bind(DeclName)))));
}

/// The pattern is very common in tests, and it is OK to use it there.
/// We have to heuristics for detecting tests: method name starts with "test"
/// (used in XCTest), and a class name contains "mock" or "test" (used in
/// helpers which are not tests themselves, but used exclusively in tests).
static bool isTest(const Decl *D) {
  if (const auto* ND = dyn_cast<NamedDecl>(D)) {
    std::string DeclName = ND->getNameAsString();
    if (StringRef(DeclName).starts_with("test"))
      return true;
  }
  if (const auto *OD = dyn_cast<ObjCMethodDecl>(D)) {
    if (const auto *CD = dyn_cast<ObjCContainerDecl>(OD->getParent())) {
      std::string ContainerName = CD->getNameAsString();
      StringRef CN(ContainerName);
      if (CN.contains_insensitive("test") || CN.contains_insensitive("mock"))
        return true;
    }
  }
  return false;
}

static auto findGCDAntiPatternWithSemaphore() -> decltype(compoundStmt()) {

  const char *SemaphoreBinding = "semaphore_name";
  auto SemaphoreCreateM = callExpr(allOf(
      callsName("dispatch_semaphore_create"),
      hasArgument(0, ignoringParenCasts(integerLiteral(equals(0))))));

  auto SemaphoreBindingM = anyOf(
      forEachDescendant(
          varDecl(hasDescendant(SemaphoreCreateM)).bind(SemaphoreBinding)),
      forEachDescendant(binaryOperator(bindAssignmentToDecl(SemaphoreBinding),
                     hasRHS(SemaphoreCreateM))));

  auto HasBlockArgumentM = hasAnyArgument(hasType(
            hasCanonicalType(blockPointerType())
            ));

  auto ArgCallsSignalM = hasAnyArgument(stmt(hasDescendant(callExpr(
          allOf(
              callsName("dispatch_semaphore_signal"),
              equalsBoundArgDecl(0, SemaphoreBinding)
              )))));

  auto HasBlockAndCallsSignalM = allOf(HasBlockArgumentM, ArgCallsSignalM);

  auto HasBlockCallingSignalM =
    forEachDescendant(
      stmt(anyOf(
        callExpr(HasBlockAndCallsSignalM),
        objcMessageExpr(HasBlockAndCallsSignalM)
           )));

  auto SemaphoreWaitM = forEachDescendant(
    callExpr(
      allOf(
        callsName("dispatch_semaphore_wait"),
        equalsBoundArgDecl(0, SemaphoreBinding)
      )
    ).bind(WarnAtNode));

  return compoundStmt(
      SemaphoreBindingM, HasBlockCallingSignalM, SemaphoreWaitM);
}

static auto findGCDAntiPatternWithGroup() -> decltype(compoundStmt()) {

  const char *GroupBinding = "group_name";
  auto DispatchGroupCreateM = callExpr(callsName("dispatch_group_create"));

  auto GroupBindingM = anyOf(
      forEachDescendant(
          varDecl(hasDescendant(DispatchGroupCreateM)).bind(GroupBinding)),
      forEachDescendant(binaryOperator(bindAssignmentToDecl(GroupBinding),
                     hasRHS(DispatchGroupCreateM))));

  auto GroupEnterM = forEachDescendant(
      stmt(callExpr(allOf(callsName("dispatch_group_enter"),
                          equalsBoundArgDecl(0, GroupBinding)))));

  auto HasBlockArgumentM = hasAnyArgument(hasType(
            hasCanonicalType(blockPointerType())
            ));

  auto ArgCallsSignalM = hasAnyArgument(stmt(hasDescendant(callExpr(
          allOf(
              callsName("dispatch_group_leave"),
              equalsBoundArgDecl(0, GroupBinding)
              )))));

  auto HasBlockAndCallsLeaveM = allOf(HasBlockArgumentM, ArgCallsSignalM);

  auto AcceptsBlockM =
    forEachDescendant(
      stmt(anyOf(
        callExpr(HasBlockAndCallsLeaveM),
        objcMessageExpr(HasBlockAndCallsLeaveM)
           )));

  auto GroupWaitM = forEachDescendant(
    callExpr(
      allOf(
        callsName("dispatch_group_wait"),
        equalsBoundArgDecl(0, GroupBinding)
      )
    ).bind(WarnAtNode));

  return compoundStmt(GroupBindingM, GroupEnterM, AcceptsBlockM, GroupWaitM);
}

static void emitDiagnostics(const BoundNodes &Nodes,
                            const char* Type,
                            BugReporter &BR,
                            AnalysisDeclContext *ADC,
                            const GCDAntipatternChecker *Checker) {
  const auto *SW = Nodes.getNodeAs<CallExpr>(WarnAtNode);
  assert(SW);

  std::string Diagnostics;
  toolchain::raw_string_ostream OS(Diagnostics);
  OS << "Waiting on a callback using a " << Type << " creates useless threads "
     << "and is subject to priority inversion; consider "
     << "using a synchronous API or changing the caller to be asynchronous";

  BR.EmitBasicReport(
    ADC->getDecl(),
    Checker,
    /*Name=*/"GCD performance anti-pattern",
    /*BugCategory=*/"Performance",
    OS.str(),
    PathDiagnosticLocation::createBegin(SW, BR.getSourceManager(), ADC),
    SW->getSourceRange());
}

void GCDAntipatternChecker::checkASTCodeBody(const Decl *D,
                                             AnalysisManager &AM,
                                             BugReporter &BR) const {
  if (isTest(D))
    return;

  AnalysisDeclContext *ADC = AM.getAnalysisDeclContext(D);

  auto SemaphoreMatcherM = findGCDAntiPatternWithSemaphore();
  auto Matches = match(SemaphoreMatcherM, *D->getBody(), AM.getASTContext());
  for (BoundNodes Match : Matches)
    emitDiagnostics(Match, "semaphore", BR, ADC, this);

  auto GroupMatcherM = findGCDAntiPatternWithGroup();
  Matches = match(GroupMatcherM, *D->getBody(), AM.getASTContext());
  for (BoundNodes Match : Matches)
    emitDiagnostics(Match, "group", BR, ADC, this);
}

} // end of anonymous namespace

void ento::registerGCDAntipattern(CheckerManager &Mgr) {
  Mgr.registerChecker<GCDAntipatternChecker>();
}

bool ento::shouldRegisterGCDAntipattern(const CheckerManager &mgr) {
  return true;
}
