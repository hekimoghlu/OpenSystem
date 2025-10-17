/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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

//===--- PassManagerVerifierAnalysis.cpp ----------------------------------===//
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

#define DEBUG_TYPE "sil-passmanager-verifier-analysis"
#include "language/SILOptimizer/Analysis/PassManagerVerifierAnalysis.h"
#include "language/Basic/Assertions.h"
#include "language/SIL/SILModule.h"
#include "toolchain/Support/CommandLine.h"

static toolchain::cl::opt<bool>
    EnableVerifier("enable-sil-passmanager-verifier-analysis",
                   toolchain::cl::desc("Enable verification of the passmanagers "
                                  "function notification infrastructure"),
                   toolchain::cl::init(true));

using namespace language;

PassManagerVerifierAnalysis::PassManagerVerifierAnalysis(SILModule *mod)
    : SILAnalysis(SILAnalysisKind::PassManagerVerifier), mod(*mod) {
#ifndef NDEBUG
  if (!EnableVerifier)
    return;
  for (auto &fn : *mod) {
    TOOLCHAIN_DEBUG(toolchain::dbgs() << "PMVerifierAnalysis. Add: " << fn.getName()
                            << '\n');
    liveFunctionNames.insert(fn.getName());
  }
#endif
}

/// Validate that the analysis is able to look up all functions and that those
/// functions are live.
void PassManagerVerifierAnalysis::invalidate() {}

/// Validate that the analysis is able to look up the given function.
void PassManagerVerifierAnalysis::invalidate(SILFunction *f,
                                             InvalidationKind k) {}

/// If a function has not yet been seen start tracking it.
void PassManagerVerifierAnalysis::notifyAddedOrModifiedFunction(
    SILFunction *f) {
#ifndef NDEBUG
  if (!EnableVerifier)
    return;
  TOOLCHAIN_DEBUG(toolchain::dbgs() << "PMVerifierAnalysis. Add|Mod: " << f->getName()
                          << '\n');
  liveFunctionNames.insert(f->getName());
#endif
}

/// Stop tracking a function.
void PassManagerVerifierAnalysis::notifyWillDeleteFunction(SILFunction *f) {
#ifndef NDEBUG
  if (!EnableVerifier)
    return;
  TOOLCHAIN_DEBUG(toolchain::dbgs() << "PMVerifierAnalysis. Delete: " << f->getName()
                          << '\n');
  if (liveFunctionNames.erase(f->getName()))
    return;

  toolchain::errs()
      << "Error! Tried to delete function that analysis was not aware of: "
      << f->getName() << '\n';
  toolchain_unreachable("triggering standard assertion failure routine");
#endif
}

/// Make sure that when we invalidate a function table, make sure we can find
/// all functions for all witness tables.
void PassManagerVerifierAnalysis::invalidateFunctionTables() {}

/// Run the entire verification.
void PassManagerVerifierAnalysis::verifyFull() const {
#ifndef NDEBUG
  if (!EnableVerifier)
    return;

  // We check that liveFunctionNames is in sync with the module's function list
  // by going through the module's function list and attempting to remove all
  // functions in the module. If we fail to remove fn, then we know that a
  // function was added to the module without an appropriate message being sent
  // by the pass manager.
  bool foundError = false;

  unsigned count = 0;
  for (auto &fn : mod) {
    if (liveFunctionNames.count(fn.getName())) {
      ++count;
      continue;
    }
    toolchain::errs() << "Found function in module that was not added to verifier: "
                 << fn.getName() << '\n';
    foundError = true;
  }

  // Ok, so now we know that function(mod) is a subset of
  // liveFunctionNames. Relying on the uniqueness provided by the module's
  // function list, we know that liveFunction should be exactly count in
  // size. Otherwise, we must have an error. If and only if we detect this
  // error, do the expensive work of finding the missing deletes. This is an
  // important performance optimization to avoid a large copy on the hot path.
  if (liveFunctionNames.size() != count) {
    auto liveFunctionNamesCopy = toolchain::StringSet<>(liveFunctionNames);
    for (auto &fn : mod) {
      liveFunctionNamesCopy.erase(fn.getName());
    }
    for (auto &iter : liveFunctionNamesCopy) {
      toolchain::errs() << "Missing delete message for function: " << iter.first()
                   << '\n';
      foundError = true;
    }
  }

  // We assert here so we emit /all/ errors before asserting.
  assert(!foundError && "triggering standard assertion failure routine");
#endif
}

//===----------------------------------------------------------------------===//
//                              Main Entry Point
//===----------------------------------------------------------------------===//

SILAnalysis *language::createPassManagerVerifierAnalysis(SILModule *m) {
  return new PassManagerVerifierAnalysis(m);
}
