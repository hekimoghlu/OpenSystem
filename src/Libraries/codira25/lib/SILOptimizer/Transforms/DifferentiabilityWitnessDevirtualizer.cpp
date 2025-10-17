/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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

//===--- DifferentiabilityWitnessDevirtualizer.cpp ------------------------===//
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
//
// Devirtualizes `differentiability_witness_function` instructions into
// `function_ref` instructions for differentiability witness definitions.
//
//===----------------------------------------------------------------------===//

#include "language/Basic/Assertions.h"
#include "language/SIL/SILBuilder.h"
#include "language/SIL/SILFunction.h"
#include "language/SIL/SILInstruction.h"
#include "language/SILOptimizer/PassManager/Transforms.h"

using namespace language;

namespace {
class DifferentiabilityWitnessDevirtualizer : public SILFunctionTransform {

  /// Returns true if any changes were made.
  bool devirtualizeDifferentiabilityWitnessesInFunction(SILFunction &f);

  /// The entry point to the transformation.
  void run() override {
    if (devirtualizeDifferentiabilityWitnessesInFunction(*getFunction()))
      invalidateAnalysis(SILAnalysis::InvalidationKind::CallsAndInstructions);
  }
};
} // end anonymous namespace

bool DifferentiabilityWitnessDevirtualizer::
    devirtualizeDifferentiabilityWitnessesInFunction(SILFunction &f) {
  bool changed = false;
  toolchain::SmallVector<DifferentiabilityWitnessFunctionInst *, 8> insts;
  for (auto &bb : f) {
    for (auto &inst : bb) {
      auto *dfwi = dyn_cast<DifferentiabilityWitnessFunctionInst>(&inst);
      if (!dfwi)
        continue;
      insts.push_back(dfwi);
    }
  }
  for (auto *inst : insts) {
    auto *witness = inst->getWitness();
    if (witness->isDeclaration())
      f.getModule().loadDifferentiabilityWitness(witness);
    if (witness->isDeclaration())
      continue;
    changed = true;
    SILBuilderWithScope builder(inst);
    auto kind = inst->getWitnessKind().getAsDerivativeFunctionKind();
    assert(kind.has_value());
    auto *newInst = builder.createFunctionRefFor(inst->getLoc(),
                                                 witness->getDerivative(*kind));
    inst->replaceAllUsesWith(newInst);
    inst->getParent()->erase(inst);
  }
  return changed;
}

SILTransform *language::createDifferentiabilityWitnessDevirtualizer() {
  return new DifferentiabilityWitnessDevirtualizer();
}
