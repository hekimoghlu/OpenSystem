/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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

//===-- AllocaHoisting.cpp - Hoist allocas to the entry block --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Hoist the alloca instructions in the non-entry blocks to the entry blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"
#include "NVPTXAllocaHoisting.h"

namespace llvm {

bool NVPTXAllocaHoisting::runOnFunction(Function &function) {
  bool               functionModified    = false;
  Function::iterator I                   = function.begin();
  TerminatorInst    *firstTerminatorInst = (I++)->getTerminator();

  for (Function::iterator E = function.end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {
      AllocaInst *allocaInst = dyn_cast<AllocaInst>(BI++);
      if (allocaInst && isa<ConstantInt>(allocaInst->getArraySize())) {
        allocaInst->moveBefore(firstTerminatorInst);
        functionModified = true;
      }
    }
  }

  return functionModified;
}

char NVPTXAllocaHoisting::ID = 1;
RegisterPass<NVPTXAllocaHoisting> X("alloca-hoisting",
                                    "Hoisting alloca instructions in non-entry "
                                    "blocks to the entry block");

FunctionPass *createAllocaHoisting() {
  return new NVPTXAllocaHoisting();
}

} // end namespace llvm
