/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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

//===- Mem2Reg.cpp - The -mem2reg pass, a wrapper around the Utils lib ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mem2reg"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumPromoted, "Number of alloca's promoted");

namespace {
  struct PromotePass : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    PromotePass() : FunctionPass(ID) {
      initializePromotePassPass(*PassRegistry::getPassRegistry());
    }

    // runOnFunction - To run this pass, first we calculate the alloca
    // instructions that are safe for promotion, then we promote each one.
    //
    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.setPreservesCFG();
      // This is a cluster of orthogonal Transforms
      AU.addPreserved<UnifyFunctionExitNodes>();
      AU.addPreservedID(LowerSwitchID);
      AU.addPreservedID(LowerInvokePassID);
    }
  };
}  // end of anonymous namespace

char PromotePass::ID = 0;
INITIALIZE_PASS_BEGIN(PromotePass, "mem2reg", "Promote Memory to Register",
                false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_END(PromotePass, "mem2reg", "Promote Memory to Register",
                false, false)

bool PromotePass::runOnFunction(Function &F) {
  std::vector<AllocaInst*> Allocas;

  BasicBlock &BB = F.getEntryBlock();  // Get the entry node for the function

  bool Changed  = false;

  DominatorTree &DT = getAnalysis<DominatorTree>();

  while (1) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I))       // Is it an alloca?
        if (isAllocaPromotable(AI))
          Allocas.push_back(AI);

    if (Allocas.empty()) break;

    PromoteMemToReg(Allocas, DT);
    NumPromoted += Allocas.size();
    Changed = true;
  }

  return Changed;
}

// createPromoteMemoryToRegister - Provide an entry point to create this pass.
//
FunctionPass *llvm::createPromoteMemoryToRegisterPass() {
  return new PromotePass();
}
