/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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

//=- llvm/CodeGen/MachineDominators.h ----------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes interfaces to post dominance information for
// target-specific code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEPOSTDOMINATORS_H
#define LLVM_CODEGEN_MACHINEPOSTDOMINATORS_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/DominatorInternals.h"

namespace llvm {

///
/// PostDominatorTree Class - Concrete subclass of DominatorTree that is used
/// to compute the a post-dominator tree.
///
struct MachinePostDominatorTree : public MachineFunctionPass {
private:
  DominatorTreeBase<MachineBasicBlock> *DT;

public:
  static char ID;

  MachinePostDominatorTree();

  ~MachinePostDominatorTree();

  FunctionPass *createMachinePostDominatorTreePass();

  const std::vector<MachineBasicBlock *> &getRoots() const {
    return DT->getRoots();
  }

  MachineDomTreeNode *getRootNode() const {
    return DT->getRootNode();
  }

  MachineDomTreeNode *operator[](MachineBasicBlock *BB) const {
    return DT->getNode(BB);
  }

  MachineDomTreeNode *getNode(MachineBasicBlock *BB) const {
    return DT->getNode(BB);
  }

  bool dominates(MachineDomTreeNode *A, MachineDomTreeNode *B) const {
    return DT->dominates(A, B);
  }

  bool dominates(MachineBasicBlock *A, MachineBasicBlock *B) const {
    return DT->dominates(A, B);
  }

  bool
  properlyDominates(const MachineDomTreeNode *A, MachineDomTreeNode *B) const {
    return DT->properlyDominates(A, B);
  }

  bool
  properlyDominates(MachineBasicBlock *A, MachineBasicBlock *B) const {
    return DT->properlyDominates(A, B);
  }

  MachineBasicBlock *findNearestCommonDominator(MachineBasicBlock *A,
                                                       MachineBasicBlock *B) {
    return DT->findNearestCommonDominator(A, B);
  }

  virtual bool runOnMachineFunction(MachineFunction &MF);
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void print(llvm::raw_ostream &OS, const Module *M = 0) const;
};
} //end of namespace llvm

#endif
