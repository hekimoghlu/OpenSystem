/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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

//===--- MoveOnlyTempAllocationFromLetTester.cpp --------------------------===//
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
/// \file A simple tester for the utility function
/// eliminateTemporaryAllocationsFromLet that allows us to write separate SIL
/// test cases for the utility.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sil-move-only-checker"

#include "MoveOnlyAddressCheckerUtils.h"
#include "MoveOnlyDiagnostics.h"
#include "MoveOnlyUtils.h"

#include "language/Basic/Assertions.h"
#include "language/SILOptimizer/PassManager/Passes.h"
#include "language/SILOptimizer/PassManager/Transforms.h"

using namespace language;
using namespace language::siloptimizer;

namespace {

struct MoveOnlyTempAllocationFromLetTester : SILFunctionTransform {
  void run() override {
    auto *fn = getFunction();

    // Don't rerun diagnostics on deserialized functions.
    if (getFunction()->wasDeserializedCanonical())
      return;

    assert(fn->getModule().getStage() == SILStage::Raw &&
           "Should only run on Raw SIL");

    TOOLCHAIN_DEBUG(toolchain::dbgs()
               << "===> MoveOnlyTempAllocationFromLetTester. Visiting: "
               << fn->getName() << '\n');

    toolchain::SmallSetVector<MarkUnresolvedNonCopyableValueInst *, 32>
        moveIntroducersToProcess;
    DiagnosticEmitter diagnosticEmitter(getFunction());

    unsigned diagCount = diagnosticEmitter.getDiagnosticCount();
    searchForCandidateAddressMarkUnresolvedNonCopyableValueInsts(
        getFunction(), getAnalysis<PostOrderAnalysis>(),
        moveIntroducersToProcess, diagnosticEmitter);

    // Return early if we emitted a diagnostic.
    if (diagCount != diagnosticEmitter.getDiagnosticCount())
      return;

    bool madeChange = false;
    while (!moveIntroducersToProcess.empty()) {
      auto *next = moveIntroducersToProcess.pop_back_val();
      madeChange |= eliminateTemporaryAllocationsFromLet(next);
    }

    if (madeChange)
      invalidateAnalysis(SILAnalysis::InvalidationKind::Instructions);
  }
};

} // namespace

SILTransform *language::createMoveOnlyTempAllocationFromLetTester() {
  return new MoveOnlyTempAllocationFromLetTester();
}
