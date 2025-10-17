/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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

//===--- MoveOnlyBorrowToDestructureTester.cpp ----------------------------===//
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
/// \file This is a pass that converts the borrow + gep pattern to destructures
/// or emits an error if it cannot be done. It is assumed that it runs
/// immediately before move checking of objects runs. This ensures that the move
/// checker does not need to worry about this problem and instead can just check
/// that the newly inserted destructures do not cause move only errors.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sil-move-only-checker"

#include "MoveOnlyBorrowToDestructureUtils.h"
#include "MoveOnlyDiagnostics.h"
#include "MoveOnlyObjectCheckerUtils.h"
#include "MoveOnlyUtils.h"

#include "language/Basic/Assertions.h"
#include "language/Basic/BlotSetVector.h"
#include "language/Basic/Defer.h"
#include "language/Basic/FrozenMultiMap.h"
#include "language/SIL/SILBuilder.h"
#include "language/SIL/SILInstruction.h"
#include "language/SILOptimizer/Analysis/Analysis.h"
#include "language/SILOptimizer/Analysis/DeadEndBlocksAnalysis.h"
#include "language/SILOptimizer/Analysis/PostOrderAnalysis.h"
#include "language/SILOptimizer/PassManager/Passes.h"
#include "language/SILOptimizer/PassManager/Transforms.h"
#include "language/SILOptimizer/Utils/CFGOptUtils.h"
#include "toolchain/ADT/ArrayRef.h"

using namespace language;
using namespace language::siloptimizer;
using namespace language::siloptimizer::borrowtodestructure;

//===----------------------------------------------------------------------===//
//                            Top Level Entrypoint
//===----------------------------------------------------------------------===//

static bool runTransform(
    SILFunction *fn,
    ArrayRef<MarkUnresolvedNonCopyableValueInst *> moveIntroducersToProcess,
    PostOrderAnalysis *poa, DiagnosticEmitter &diagnosticEmitter) {
  IntervalMapAllocator allocator;
  bool madeChange = false;
  while (!moveIntroducersToProcess.empty()) {
    auto *mmci = moveIntroducersToProcess.back();
    moveIntroducersToProcess = moveIntroducersToProcess.drop_back();

    BorrowToDestructureTransform transform(allocator, mmci, mmci,
                                           diagnosticEmitter, poa);
    transform.transform();
    madeChange = true;
  }

  return madeChange;
}

namespace {

class MoveOnlyBorrowToDestructureTransformPass : public SILFunctionTransform {
  void run() override {
    auto *fn = getFunction();

    // Don't rerun diagnostics on deserialized functions.
    if (getFunction()->wasDeserializedCanonical())
      return;

    assert(fn->getModule().getStage() == SILStage::Raw &&
           "Should only run on Raw SIL");

    TOOLCHAIN_DEBUG(toolchain::dbgs()
               << "===> MoveOnlyBorrowToDestructureTransform. Visiting: "
               << fn->getName() << '\n');

    auto *postOrderAnalysis = getAnalysis<PostOrderAnalysis>();

    toolchain::SmallSetVector<MarkUnresolvedNonCopyableValueInst *, 32>
        moveIntroducersToProcess;
    DiagnosticEmitter diagnosticEmitter(getFunction());

    unsigned diagCount = diagnosticEmitter.getDiagnosticCount();
    bool madeChange =
        searchForCandidateObjectMarkUnresolvedNonCopyableValueInsts(
            getFunction(), moveIntroducersToProcess, diagnosticEmitter);
    if (madeChange) {
      invalidateAnalysis(SILAnalysis::InvalidationKind::Instructions);
    }

    if (diagCount != diagnosticEmitter.getDiagnosticCount()) {
      if (cleanupNonCopyableCopiesAfterEmittingDiagnostic(fn))
        invalidateAnalysis(SILAnalysis::InvalidationKind::Instructions);
      return;
    }

    diagCount = diagnosticEmitter.getDiagnosticCount();
    auto introducers = toolchain::ArrayRef(moveIntroducersToProcess.begin(),
                                      moveIntroducersToProcess.end());
    if (runTransform(fn, introducers, postOrderAnalysis, diagnosticEmitter)) {
      invalidateAnalysis(SILAnalysis::InvalidationKind::Instructions);
    }

    if (diagCount != diagnosticEmitter.getDiagnosticCount()) {
      if (cleanupNonCopyableCopiesAfterEmittingDiagnostic(fn))
        invalidateAnalysis(SILAnalysis::InvalidationKind::Instructions);
    }
  }
};

} // namespace

SILTransform *language::createMoveOnlyBorrowToDestructureTransform() {
  return new MoveOnlyBorrowToDestructureTransformPass();
}
