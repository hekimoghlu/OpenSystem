/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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

//===- NVPTXSplitBBatBar.cpp - Split BB at Barrier  --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Split basic blocks so that a basic block that contains a barrier instruction
// only contains the barrier instruction.
//
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Support/InstIterator.h"
#include "NVPTXUtilities.h"
#include "NVPTXSplitBBatBar.h"

using namespace llvm;

namespace llvm {
FunctionPass *createSplitBBatBarPass();
}

char NVPTXSplitBBatBar::ID = 0;

bool NVPTXSplitBBatBar::runOnFunction(Function &F) {

  SmallVector<Instruction *, 4> SplitPoints;
  bool changed = false;

  // Collect all the split points in SplitPoints
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI) {
    BasicBlock::iterator IB = BI->begin();
    BasicBlock::iterator II = IB;
    BasicBlock::iterator IE = BI->end();

    // Skit the first intruction. No splitting is needed at this
    // point even if this is a bar.
    while (II != IE) {
      if (IntrinsicInst *inst = dyn_cast<IntrinsicInst>(II)) {
        Intrinsic::ID id = inst->getIntrinsicID();
        // If this is a barrier, split at this instruction
        // and the next instruction.
        if (llvm::isBarrierIntrinsic(id)) {
          if (II != IB)
            SplitPoints.push_back(II);
          II++;
          if ((II != IE) && (!II->isTerminator())) {
            SplitPoints.push_back(II);
            II++;
          }
          continue;
        }
      }
      II++;
    }
  }

  for (unsigned i = 0; i != SplitPoints.size(); i++) {
    changed = true;
    Instruction *inst = SplitPoints[i];
    inst->getParent()->splitBasicBlock(inst, "bar_split");
  }

  return changed;
}

// This interface will most likely not be necessary, because this pass will
// not be invoked by the driver, but will be used as a prerequisite to
// another pass.
FunctionPass *llvm::createSplitBBatBarPass() {
  return new NVPTXSplitBBatBar();
}
