/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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

//===--- SimplifyUnreachableContainingBlocks.cpp --------------------------===//
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
/// This file contains a simple utility pass that simplifies blocks that contain
/// unreachables by eliminating all other instructions. This includes
/// instructions with side-effects and no-return functions. It is only intended
/// to be used to simplify IR for testing or exploratory purposes.
///
//===----------------------------------------------------------------------===//

#include "language/SILOptimizer/PassManager/Passes.h"
#include "language/SIL/SILBasicBlock.h"
#include "language/SILOptimizer/PassManager/Transforms.h"

using namespace language;

namespace {

class SimplifyUnreachableContainingBlocks : public SILFunctionTransform {
  void run() override {
    // For each block...
    for (auto &BB : *getFunction()) {
      // If the block does not contain an unreachable, just continue. There is
      // no further work to do.
      auto *UI = dyn_cast<UnreachableInst>(BB.getTerminator());
      if (!UI)
        continue;

      // Otherwise, eliminate all other instructions in the block.
      for (auto II = BB.begin(); &*II != UI;) {
        // Avoid iterator invalidation.
        auto *I = &*II;
        ++II;

        I->replaceAllUsesOfAllResultsWithUndef();
        I->eraseFromParent();
      }
    }
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//                           Top Level Entry Point
//===----------------------------------------------------------------------===//

SILTransform *language::createSimplifyUnreachableContainingBlocks() {
  return new SimplifyUnreachableContainingBlocks();
}
