/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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

//===--- StripDebugInfo.cpp - Strip debug info from SIL -------------------===//
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

#include "language/SILOptimizer/PassManager/Passes.h"
#include "language/SILOptimizer/PassManager/Transforms.h"
#include "language/SIL/SILInstruction.h"
#include "language/SIL/SILModule.h"
#include "language/SIL/SILFunction.h"


using namespace language;

static void stripFunction(SILFunction *F) {
  for (auto &BB : *F)
    for (auto II = BB.begin(), IE = BB.end(); II != IE;) {
      SILInstruction *Inst = &*II;
      ++II;

      if (!isa<DebugValueInst>(Inst))
        continue;

      Inst->eraseFromParent();
    }
}

namespace {
class StripDebugInfo : public language::SILFunctionTransform {
  ~StripDebugInfo() override {}

  /// The entry point to the transformation.
  void run() override {
    stripFunction(getFunction());
    invalidateAnalysis(SILAnalysis::InvalidationKind::Instructions);
  }

};
} // end anonymous namespace


SILTransform *language::createStripDebugInfo() {
  return new StripDebugInfo();
}
