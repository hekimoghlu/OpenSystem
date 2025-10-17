/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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

//===--- ARCLoopOpts.cpp --------------------------------------------------===//
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
///
/// \file
///
/// This is a pass that runs multiple interrelated loop passes on a function. It
/// also provides caching of certain analysis information that is used by all of
/// the passes.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arc-sequence-opts"
#include "language/SILOptimizer/PassManager/Passes.h"
#include "ARCSequenceOpts.h"
#include "language/SILOptimizer/Analysis/AliasAnalysis.h"
#include "language/SILOptimizer/Analysis/DominanceAnalysis.h"
#include "language/SILOptimizer/Analysis/LoopAnalysis.h"
#include "language/SILOptimizer/Analysis/LoopRegionAnalysis.h"
#include "language/SILOptimizer/Analysis/ProgramTerminationAnalysis.h"
#include "language/SILOptimizer/Analysis/RCIdentityAnalysis.h"
#include "language/SILOptimizer/PassManager/Transforms.h"

using namespace language;

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

class ARCLoopOpts : public SILFunctionTransform {

  void run() override {
    auto *F = getFunction();

    // If ARC optimizations are disabled, don't optimize anything and bail.
    if (!getOptions().EnableARCOptimizations)
      return;

    // Skip global init functions.
    if (F->isGlobalInitOnceFunction())
      return;

    auto *LA = getAnalysis<SILLoopAnalysis>();
    auto *LI = LA->get(F);
    auto *DA = getAnalysis<DominanceAnalysis>();
    auto *DI = DA->get(F);

    // Canonicalize the loops, invalidating if we need to.
    if (canonicalizeAllLoops(DI, LI)) {
      // We preserve loop info and the dominator tree.
      DA->lockInvalidation();
      LA->lockInvalidation();
      PM->invalidateAnalysis(F, SILAnalysis::InvalidationKind::FunctionBody);
      DA->unlockInvalidation();
      LA->unlockInvalidation();
    }

    // Get all of the analyses that we need.
    auto *AA = getAnalysis<AliasAnalysis>(F);
    auto *RCFI = getAnalysis<RCIdentityAnalysis>()->get(F);
    auto *EAFI = getAnalysis<EpilogueARCAnalysis>()->get(F);
    auto *LRFI = getAnalysis<LoopRegionAnalysis>()->get(F);
    ProgramTerminationFunctionInfo PTFI(F);

    // Create all of our visitors, register them with the visitor group, and
    // run.
    LoopARCPairingContext LoopARCContext(*F, AA, LRFI, LI, RCFI, EAFI, &PTFI);
    SILLoopVisitorGroup VisitorGroup(F, LI);
    VisitorGroup.addVisitor(&LoopARCContext);
    VisitorGroup.run();

    if (LoopARCContext.madeChange()) {
      invalidateAnalysis(SILAnalysis::InvalidationKind::CallsAndInstructions);
    }
  }
};

} // end anonymous namespace

SILTransform *language::createARCLoopOpts() {
  return new ARCLoopOpts();
}
