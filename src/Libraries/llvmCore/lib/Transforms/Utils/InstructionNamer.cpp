/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 26, 2025.
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

//===- InstructionNamer.cpp - Give anonymous instructions names -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a little utility pass that gives instructions names, this is mostly
// useful when diffing the effect of an optimization because deleting an
// unnamed instruction can change all other instruction numbering, making the
// diff very noisy.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
using namespace llvm;

namespace {
  struct InstNamer : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    InstNamer() : FunctionPass(ID) {
      initializeInstNamerPass(*PassRegistry::getPassRegistry());
    }
    
    void getAnalysisUsage(AnalysisUsage &Info) const {
      Info.setPreservesAll();
    }

    bool runOnFunction(Function &F) {
      for (Function::arg_iterator AI = F.arg_begin(), AE = F.arg_end();
           AI != AE; ++AI)
        if (!AI->hasName() && !AI->getType()->isVoidTy())
          AI->setName("arg");

      for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
        if (!BB->hasName())
          BB->setName("bb");
        
        for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
          if (!I->hasName() && !I->getType()->isVoidTy())
            I->setName("tmp");
      }
      return true;
    }
  };
  
  char InstNamer::ID = 0;
}

INITIALIZE_PASS(InstNamer, "instnamer", 
                "Assign names to anonymous instructions", false, false)
char &llvm::InstructionNamerID = InstNamer::ID;
//===----------------------------------------------------------------------===//
//
// InstructionNamer - Give any unnamed non-void instructions "tmp" names.
//
FunctionPass *llvm::createInstructionNamerPass() {
  return new InstNamer();
}
