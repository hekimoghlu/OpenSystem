/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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

#include "language/Core/StaticAnalyzer/Core/Checker.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "language/Core/StaticAnalyzer/Frontend/CheckerRegistry.h"

using namespace language::Core;
using namespace ento;

// This barebones plugin is used by clang/test/Analysis/checker-plugins.c
// to test option handling on checkers loaded from plugins.

namespace {
struct MyChecker : public Checker<check::BeginFunction> {
  void checkBeginFunction(CheckerContext &Ctx) const {}
};

void registerMyChecker(CheckerManager &Mgr) {
  MyChecker *Checker = Mgr.registerChecker<MyChecker>();
  toolchain::outs() << "Example option is set to "
               << (Mgr.getAnalyzerOptions().getCheckerBooleanOption(
                       Checker, "ExampleOption")
                       ? "true"
                       : "false")
               << '\n';
}

bool shouldRegisterMyChecker(const CheckerManager &mgr) { return true; }

} // end anonymous namespace

// Register plugin!
extern "C" void clang_registerCheckers(CheckerRegistry &Registry) {
  Registry.addChecker(registerMyChecker, shouldRegisterMyChecker,
                      "example.MyChecker", "Example Description");

  Registry.addCheckerOption(/*OptionType*/ "bool",
                            /*CheckerFullName*/ "example.MyChecker",
                            /*OptionName*/ "ExampleOption",
                            /*DefaultValStr*/ "false",
                            /*Description*/ "This is an example checker opt.",
                            /*DevelopmentStage*/ "released");
}

extern "C" const char clang_analyzerAPIVersionString[] =
    CLANG_ANALYZER_API_VERSION_STRING;
