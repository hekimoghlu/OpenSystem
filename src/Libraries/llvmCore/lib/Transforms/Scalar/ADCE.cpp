/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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

//===- DCE.cpp - Code to perform dead code elimination --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Aggressive Dead Code Elimination pass.  This pass
// optimistically assumes that all instructions are dead until proven otherwise,
// allowing it to eliminate dead computations that other DCE passes do not
// catch, particularly involving loop computations.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "adce"
#include "llvm/Transforms/Scalar.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumRemoved, "Number of instructions removed");

namespace {
  struct ADCE : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    ADCE() : FunctionPass(ID) {
      initializeADCEPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function& F);

    virtual void getAnalysisUsage(AnalysisUsage& AU) const {
      AU.setPreservesCFG();
    }

  };
}

char ADCE::ID = 0;
INITIALIZE_PASS(ADCE, "adce", "Aggressive Dead Code Elimination", false, false)

bool ADCE::runOnFunction(Function& F) {
  SmallPtrSet<Instruction*, 128> alive;
  SmallVector<Instruction*, 128> worklist;

  // Collect the set of "root" instructions that are known live.
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
    if (isa<TerminatorInst>(I.getInstructionIterator()) ||
        isa<DbgInfoIntrinsic>(I.getInstructionIterator()) ||
        isa<LandingPadInst>(I.getInstructionIterator()) ||
        I->mayHaveSideEffects()) {
      alive.insert(I.getInstructionIterator());
      worklist.push_back(I.getInstructionIterator());
    }

  // Propagate liveness backwards to operands.
  while (!worklist.empty()) {
    Instruction* curr = worklist.pop_back_val();
    for (Instruction::op_iterator OI = curr->op_begin(), OE = curr->op_end();
         OI != OE; ++OI)
      if (Instruction* Inst = dyn_cast<Instruction>(OI))
        if (alive.insert(Inst))
          worklist.push_back(Inst);
  }

  // The inverse of the live set is the dead set.  These are those instructions
  // which have no side effects and do not influence the control flow or return
  // value of the function, and may therefore be deleted safely.
  // NOTE: We reuse the worklist vector here for memory efficiency.
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
    if (!alive.count(I.getInstructionIterator())) {
      worklist.push_back(I.getInstructionIterator());
      I->dropAllReferences();
    }

  for (SmallVector<Instruction*, 1024>::iterator I = worklist.begin(),
       E = worklist.end(); I != E; ++I) {
    ++NumRemoved;
    (*I)->eraseFromParent();
  }

  return !worklist.empty();
}

FunctionPass *llvm::createAggressiveDCEPass() {
  return new ADCE();
}
