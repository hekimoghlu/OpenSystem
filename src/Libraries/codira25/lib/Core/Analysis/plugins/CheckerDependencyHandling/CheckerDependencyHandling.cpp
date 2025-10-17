/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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

// This barebones plugin is used by clang/test/Analysis/checker-plugins.c
// to test dependency handling among checkers loaded from plugins.

using namespace language::Core;
using namespace ento;

namespace {
struct Dependency : public Checker<check::BeginFunction> {
  void checkBeginFunction(CheckerContext &Ctx) const {}
};
struct DependendentChecker : public Checker<check::BeginFunction> {
  void checkBeginFunction(CheckerContext &Ctx) const {}
};
} // end anonymous namespace

// Register plugin!
extern "C" void clang_registerCheckers(CheckerRegistry &Registry) {
  Registry.addChecker<Dependency>("example.Dependency", "MockDescription");
  Registry.addChecker<DependendentChecker>("example.DependendentChecker",
                                           "MockDescription");

  Registry.addDependency("example.DependendentChecker", "example.Dependency");
}

extern "C" const char clang_analyzerAPIVersionString[] =
    CLANG_ANALYZER_API_VERSION_STRING;
