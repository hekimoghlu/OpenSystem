/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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

//===- TestPasses.cpp - "buggy" passes used to test bugpoint --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains "buggy" passes that are used to test bugpoint, to check
// that it is narrowing down testcases correctly.
//
//===----------------------------------------------------------------------===//

#include "llvm/BasicBlock.h"
#include "llvm/Constant.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Support/InstVisitor.h"

using namespace llvm;

namespace {
  /// CrashOnCalls - This pass is used to test bugpoint.  It intentionally
  /// crashes on any call instructions.
  class CrashOnCalls : public BasicBlockPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    CrashOnCalls() : BasicBlockPass(ID) {}
  private:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    bool runOnBasicBlock(BasicBlock &BB) {
      for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
        if (isa<CallInst>(*I))
          abort();

      return false;
    }
  };
}

char CrashOnCalls::ID = 0;
static RegisterPass<CrashOnCalls>
  X("bugpoint-crashcalls",
    "BugPoint Test Pass - Intentionally crash on CallInsts");

namespace {
  /// DeleteCalls - This pass is used to test bugpoint.  It intentionally
  /// deletes some call instructions, "misoptimizing" the program.
  class DeleteCalls : public BasicBlockPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    DeleteCalls() : BasicBlockPass(ID) {}
  private:
    bool runOnBasicBlock(BasicBlock &BB) {
      for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
        if (CallInst *CI = dyn_cast<CallInst>(I)) {
          if (!CI->use_empty())
            CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
          CI->getParent()->getInstList().erase(CI);
          break;
        }
      return false;
    }
  };
}
 
char DeleteCalls::ID = 0;
static RegisterPass<DeleteCalls>
  Y("bugpoint-deletecalls",
    "BugPoint Test Pass - Intentionally 'misoptimize' CallInsts");
