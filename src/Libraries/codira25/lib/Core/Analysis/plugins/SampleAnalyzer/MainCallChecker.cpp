/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

#include "language/Core/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "language/Core/StaticAnalyzer/Core/Checker.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "language/Core/StaticAnalyzer/Frontend/CheckerRegistry.h"

// This simple plugin is used by clang/test/Analysis/checker-plugins.c
// to test the use of a checker that is defined in a plugin.

using namespace language::Core;
using namespace ento;

namespace {
class MainCallChecker : public Checker<check::PreStmt<CallExpr>> {

  const BugType BT{this, "call to main", "example analyzer plugin"};

public:
  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const;
};
} // end anonymous namespace

void MainCallChecker::checkPreStmt(const CallExpr *CE,
                                   CheckerContext &C) const {
  const Expr *Callee = CE->getCallee();
  const FunctionDecl *FD = C.getSVal(Callee).getAsFunctionDecl();

  if (!FD)
    return;

  // Get the name of the callee.
  IdentifierInfo *II = FD->getIdentifier();
  if (!II) // if no identifier, not a simple C function
    return;

  if (II->isStr("main")) {
    ExplodedNode *N = C.generateErrorNode();
    if (!N)
      return;

    auto report =
        std::make_unique<PathSensitiveBugReport>(BT, BT.getDescription(), N);
    report->addRange(Callee->getSourceRange());
    C.emitReport(std::move(report));
  }
}

// Register plugin!
extern "C" void clang_registerCheckers(CheckerRegistry &Registry) {
  Registry.addChecker<MainCallChecker>("example.MainCallChecker",
                                       "Example Description");
}

extern "C" const char clang_analyzerAPIVersionString[] =
    CLANG_ANALYZER_API_VERSION_STRING;
