/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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

//===- Reg2Mem.cpp - Convert registers to allocas -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file demotes all registers to memory references.  It is intended to be
// the inverse of PromoteMemoryToRegister.  By converting to loads, the only
// values live across basic blocks are allocas and loads before phi nodes.
// It is intended that this should make CFG hacking much easier.
// To make later hacking easier, the entry block is split into two, such that
// all introduced allocas and nothing else are in the entry block.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reg2mem"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CFG.h"
#include <list>
using namespace llvm;

STATISTIC(NumRegsDemoted, "Number of registers demoted");
STATISTIC(NumPhisDemoted, "Number of phi-nodes demoted");

namespace {
  struct RegToMem : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    RegToMem() : FunctionPass(ID) {
      initializeRegToMemPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addPreservedID(BreakCriticalEdgesID);
    }

   bool valueEscapes(const Instruction *Inst) const {
     const BasicBlock *BB = Inst->getParent();
      for (Value::const_use_iterator UI = Inst->use_begin(),E = Inst->use_end();
           UI != E; ++UI) {
        const Instruction *I = cast<Instruction>(*UI);
        if (I->getParent() != BB || isa<PHINode>(I))
          return true;
      }
      return false;
    }

    virtual bool runOnFunction(Function &F);
  };
}

char RegToMem::ID = 0;
INITIALIZE_PASS_BEGIN(RegToMem, "reg2mem", "Demote all values to stack slots",
                false, false)
INITIALIZE_PASS_DEPENDENCY(BreakCriticalEdges)
INITIALIZE_PASS_END(RegToMem, "reg2mem", "Demote all values to stack slots",
                false, false)

bool RegToMem::runOnFunction(Function &F) {
  if (F.isDeclaration())
    return false;

  // Insert all new allocas into entry block.
  BasicBlock *BBEntry = &F.getEntryBlock();
  assert(pred_begin(BBEntry) == pred_end(BBEntry) &&
         "Entry block to function must not have predecessors!");

  // Find first non-alloca instruction and create insertion point. This is
  // safe if block is well-formed: it always have terminator, otherwise
  // we'll get and assertion.
  BasicBlock::iterator I = BBEntry->begin();
  while (isa<AllocaInst>(I)) ++I;

  CastInst *AllocaInsertionPoint =
    new BitCastInst(Constant::getNullValue(Type::getInt32Ty(F.getContext())),
                    Type::getInt32Ty(F.getContext()),
                    "reg2mem alloca point", I);

  // Find the escaped instructions. But don't create stack slots for
  // allocas in entry block.
  std::list<Instruction*> WorkList;
  for (Function::iterator ibb = F.begin(), ibe = F.end();
       ibb != ibe; ++ibb)
    for (BasicBlock::iterator iib = ibb->begin(), iie = ibb->end();
         iib != iie; ++iib) {
      if (!(isa<AllocaInst>(iib) && iib->getParent() == BBEntry) &&
          valueEscapes(iib)) {
        WorkList.push_front(&*iib);
      }
    }

  // Demote escaped instructions
  NumRegsDemoted += WorkList.size();
  for (std::list<Instruction*>::iterator ilb = WorkList.begin(),
       ile = WorkList.end(); ilb != ile; ++ilb)
    DemoteRegToStack(**ilb, false, AllocaInsertionPoint);

  WorkList.clear();

  // Find all phi's
  for (Function::iterator ibb = F.begin(), ibe = F.end();
       ibb != ibe; ++ibb)
    for (BasicBlock::iterator iib = ibb->begin(), iie = ibb->end();
         iib != iie; ++iib)
      if (isa<PHINode>(iib))
        WorkList.push_front(&*iib);

  // Demote phi nodes
  NumPhisDemoted += WorkList.size();
  for (std::list<Instruction*>::iterator ilb = WorkList.begin(),
       ile = WorkList.end(); ilb != ile; ++ilb)
    DemotePHIToStack(cast<PHINode>(*ilb), AllocaInsertionPoint);

  return true;
}


// createDemoteRegisterToMemory - Provide an entry point to create this pass.
char &llvm::DemoteRegisterToMemoryID = RegToMem::ID;
FunctionPass *llvm::createDemoteRegisterToMemoryPass() {
  return new RegToMem();
}
