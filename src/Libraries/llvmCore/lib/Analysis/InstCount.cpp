/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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

//===-- InstCount.cpp - Collects the count of all instructions ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass collects the count of all instructions and reports them
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instcount"
#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(TotalInsts , "Number of instructions (of all types)");
STATISTIC(TotalBlocks, "Number of basic blocks");
STATISTIC(TotalFuncs , "Number of non-external functions");
STATISTIC(TotalMemInst, "Number of memory instructions");

#define HANDLE_INST(N, OPCODE, CLASS) \
  STATISTIC(Num ## OPCODE ## Inst, "Number of " #OPCODE " insts");

#include "llvm/Instruction.def"


namespace {
  class InstCount : public FunctionPass, public InstVisitor<InstCount> {
    friend class InstVisitor<InstCount>;

    void visitFunction  (Function &F) { ++TotalFuncs; }
    void visitBasicBlock(BasicBlock &BB) { ++TotalBlocks; }

#define HANDLE_INST(N, OPCODE, CLASS) \
    void visit##OPCODE(CLASS &) { ++Num##OPCODE##Inst; ++TotalInsts; }

#include "llvm/Instruction.def"

    void visitInstruction(Instruction &I) {
      errs() << "Instruction Count does not know about " << I;
      llvm_unreachable(0);
    }
  public:
    static char ID; // Pass identification, replacement for typeid
    InstCount() : FunctionPass(ID) {
      initializeInstCountPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
    virtual void print(raw_ostream &O, const Module *M) const {}

  };
}

char InstCount::ID = 0;
INITIALIZE_PASS(InstCount, "instcount",
                "Counts the various types of Instructions", false, true)

FunctionPass *llvm::createInstCountPass() { return new InstCount(); }

// InstCount::run - This is the main Analysis entry point for a
// function.
//
bool InstCount::runOnFunction(Function &F) {
  unsigned StartMemInsts =
    NumGetElementPtrInst + NumLoadInst + NumStoreInst + NumCallInst +
    NumInvokeInst + NumAllocaInst;
  visit(F);
  unsigned EndMemInsts =
    NumGetElementPtrInst + NumLoadInst + NumStoreInst + NumCallInst +
    NumInvokeInst + NumAllocaInst;
  TotalMemInst += EndMemInsts-StartMemInsts;
  return false;
}
