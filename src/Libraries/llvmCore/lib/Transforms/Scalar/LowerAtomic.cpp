/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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

//===- LowerAtomic.cpp - Lower atomic intrinsics --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers atomic intrinsics to non-atomic form for use in a known
// non-preemptible environment.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loweratomic"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
using namespace llvm;

static bool LowerAtomicCmpXchgInst(AtomicCmpXchgInst *CXI) {
  IRBuilder<> Builder(CXI->getParent(), CXI);
  Value *Ptr = CXI->getPointerOperand();
  Value *Cmp = CXI->getCompareOperand();
  Value *Val = CXI->getNewValOperand();

  LoadInst *Orig = Builder.CreateLoad(Ptr);
  Value *Equal = Builder.CreateICmpEQ(Orig, Cmp);
  Value *Res = Builder.CreateSelect(Equal, Val, Orig);
  Builder.CreateStore(Res, Ptr);

  CXI->replaceAllUsesWith(Orig);
  CXI->eraseFromParent();
  return true;
}

static bool LowerAtomicRMWInst(AtomicRMWInst *RMWI) {
  IRBuilder<> Builder(RMWI->getParent(), RMWI);
  Value *Ptr = RMWI->getPointerOperand();
  Value *Val = RMWI->getValOperand();

  LoadInst *Orig = Builder.CreateLoad(Ptr);
  Value *Res = NULL;

  switch (RMWI->getOperation()) {
  default: llvm_unreachable("Unexpected RMW operation");
  case AtomicRMWInst::Xchg:
    Res = Val;
    break;
  case AtomicRMWInst::Add:
    Res = Builder.CreateAdd(Orig, Val);
    break;
  case AtomicRMWInst::Sub:
    Res = Builder.CreateSub(Orig, Val);
    break;
  case AtomicRMWInst::And:
    Res = Builder.CreateAnd(Orig, Val);
    break;
  case AtomicRMWInst::Nand:
    Res = Builder.CreateNot(Builder.CreateAnd(Orig, Val));
    break;
  case AtomicRMWInst::Or:
    Res = Builder.CreateOr(Orig, Val);
    break;
  case AtomicRMWInst::Xor:
    Res = Builder.CreateXor(Orig, Val);
    break;
  case AtomicRMWInst::Max:
    Res = Builder.CreateSelect(Builder.CreateICmpSLT(Orig, Val),
                               Val, Orig);
    break;
  case AtomicRMWInst::Min:
    Res = Builder.CreateSelect(Builder.CreateICmpSLT(Orig, Val),
                               Orig, Val);
    break;
  case AtomicRMWInst::UMax:
    Res = Builder.CreateSelect(Builder.CreateICmpULT(Orig, Val),
                               Val, Orig);
    break;
  case AtomicRMWInst::UMin:
    Res = Builder.CreateSelect(Builder.CreateICmpULT(Orig, Val),
                               Orig, Val);
    break;
  }
  Builder.CreateStore(Res, Ptr);
  RMWI->replaceAllUsesWith(Orig);
  RMWI->eraseFromParent();
  return true;
}

static bool LowerFenceInst(FenceInst *FI) {
  FI->eraseFromParent();
  return true;
}

static bool LowerLoadInst(LoadInst *LI) {
  LI->setAtomic(NotAtomic);
  return true;
}

static bool LowerStoreInst(StoreInst *SI) {
  SI->setAtomic(NotAtomic);
  return true;
}

namespace {
  struct LowerAtomic : public BasicBlockPass {
    static char ID;
    LowerAtomic() : BasicBlockPass(ID) {
      initializeLowerAtomicPass(*PassRegistry::getPassRegistry());
    }
    bool runOnBasicBlock(BasicBlock &BB) {
      bool Changed = false;
      for (BasicBlock::iterator DI = BB.begin(), DE = BB.end(); DI != DE; ) {
        Instruction *Inst = DI++;
        if (FenceInst *FI = dyn_cast<FenceInst>(Inst))
          Changed |= LowerFenceInst(FI);
        else if (AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(Inst))
          Changed |= LowerAtomicCmpXchgInst(CXI);
        else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(Inst))
          Changed |= LowerAtomicRMWInst(RMWI);
        else if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
          if (LI->isAtomic())
            LowerLoadInst(LI);
        } else if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
          if (SI->isAtomic())
            LowerStoreInst(SI);
        }
      }
      return Changed;
    }
  };
}

char LowerAtomic::ID = 0;
INITIALIZE_PASS(LowerAtomic, "loweratomic",
                "Lower atomic intrinsics to non-atomic form",
                false, false)

Pass *llvm::createLowerAtomicPass() { return new LowerAtomic(); }
