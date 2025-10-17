/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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

//===- LazyValueInfo.h - Value constraint analysis --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for lazy computation of value constraint
// information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LAZYVALUEINFO_H
#define LLVM_ANALYSIS_LAZYVALUEINFO_H

#include "llvm/Pass.h"

namespace llvm {
  class Constant;
  class TargetData;
  class TargetLibraryInfo;
  class Value;
  
/// LazyValueInfo - This pass computes, caches, and vends lazy value constraint
/// information.
class LazyValueInfo : public FunctionPass {
  class TargetData *TD;
  class TargetLibraryInfo *TLI;
  void *PImpl;
  LazyValueInfo(const LazyValueInfo&) LLVM_DELETED_FUNCTION;
  void operator=(const LazyValueInfo&) LLVM_DELETED_FUNCTION;
public:
  static char ID;
  LazyValueInfo() : FunctionPass(ID), PImpl(0) {
    initializeLazyValueInfoPass(*PassRegistry::getPassRegistry());
  }
  ~LazyValueInfo() { assert(PImpl == 0 && "releaseMemory not called"); }

  /// Tristate - This is used to return true/false/dunno results.
  enum Tristate {
    Unknown = -1, False = 0, True = 1
  };
  
  
  // Public query interface.
  
  /// getPredicateOnEdge - Determine whether the specified value comparison
  /// with a constant is known to be true or false on the specified CFG edge.
  /// Pred is a CmpInst predicate.
  Tristate getPredicateOnEdge(unsigned Pred, Value *V, Constant *C,
                              BasicBlock *FromBB, BasicBlock *ToBB);
  
  
  /// getConstant - Determine whether the specified value is known to be a
  /// constant at the end of the specified block.  Return null if not.
  Constant *getConstant(Value *V, BasicBlock *BB);

  /// getConstantOnEdge - Determine whether the specified value is known to be a
  /// constant on the specified edge.  Return null if not.
  Constant *getConstantOnEdge(Value *V, BasicBlock *FromBB, BasicBlock *ToBB);
  
  /// threadEdge - Inform the analysis cache that we have threaded an edge from
  /// PredBB to OldSucc to be from PredBB to NewSucc instead.
  void threadEdge(BasicBlock *PredBB, BasicBlock *OldSucc, BasicBlock *NewSucc);
  
  /// eraseBlock - Inform the analysis cache that we have erased a block.
  void eraseBlock(BasicBlock *BB);
  
  // Implementation boilerplate.
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory();
  virtual bool runOnFunction(Function &F);
};

}  // end namespace llvm

#endif

