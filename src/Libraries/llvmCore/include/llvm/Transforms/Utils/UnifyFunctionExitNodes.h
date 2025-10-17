/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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

//===-- UnifyFunctionExitNodes.h - Ensure fn's have one return --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is used to ensure that functions have at most one return and one
// unwind instruction in them.  Additionally, it keeps track of which node is
// the new exit node of the CFG.  If there are no return or unwind instructions
// in the function, the getReturnBlock/getUnwindBlock methods will return a null
// pointer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UNIFYFUNCTIONEXITNODES_H
#define LLVM_TRANSFORMS_UNIFYFUNCTIONEXITNODES_H

#include "llvm/Pass.h"

namespace llvm {

struct UnifyFunctionExitNodes : public FunctionPass {
  BasicBlock *ReturnBlock, *UnwindBlock, *UnreachableBlock;
public:
  static char ID; // Pass identification, replacement for typeid
  UnifyFunctionExitNodes() : FunctionPass(ID),
                             ReturnBlock(0), UnwindBlock(0) {
    initializeUnifyFunctionExitNodesPass(*PassRegistry::getPassRegistry());
  }

  // We can preserve non-critical-edgeness when we unify function exit nodes
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  // getReturn|Unwind|UnreachableBlock - Return the new single (or nonexistant)
  // return, unwind, or unreachable  basic blocks in the CFG.
  //
  BasicBlock *getReturnBlock() const { return ReturnBlock; }
  BasicBlock *getUnwindBlock() const { return UnwindBlock; }
  BasicBlock *getUnreachableBlock() const { return UnreachableBlock; }

  virtual bool runOnFunction(Function &F);
};

Pass *createUnifyFunctionExitNodesPass();

} // End llvm namespace

#endif
