/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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

//===-- AllocaHoisting.h - Hosist allocas to the entry block ----*- C++ -*-===//
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

#ifndef NVPTX_ALLOCA_HOISTING_H_
#define NVPTX_ALLOCA_HOISTING_H_

#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetData.h"

namespace llvm {

class FunctionPass;
class Function;

// Hoisting the alloca instructions in the non-entry blocks to the entry
// block.
class NVPTXAllocaHoisting : public FunctionPass {
public:
  static char ID; // Pass ID
  NVPTXAllocaHoisting() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<TargetData>();
    AU.addPreserved<MachineFunctionAnalysis>();
  }

  virtual const char *getPassName() const {
    return "NVPTX specific alloca hoisting";
  }

  virtual bool runOnFunction(Function &function);
};

extern FunctionPass *createAllocaHoisting();

} // end namespace llvm

#endif // NVPTX_ALLOCA_HOISTING_H_
