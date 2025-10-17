/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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

//===- MachinePostDominators.cpp -Machine Post Dominator Calculation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements simple dominator construction algorithms for finding
// post dominators on machine functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachinePostDominators.h"

using namespace llvm;

char MachinePostDominatorTree::ID = 0;

//declare initializeMachinePostDominatorTreePass
INITIALIZE_PASS(MachinePostDominatorTree, "machinepostdomtree",
                "MachinePostDominator Tree Construction", true, true)

MachinePostDominatorTree::MachinePostDominatorTree() : MachineFunctionPass(ID) {
  initializeMachinePostDominatorTreePass(*PassRegistry::getPassRegistry());
  DT = new DominatorTreeBase<MachineBasicBlock>(true); //true indicate
                                                       // postdominator
}

FunctionPass *
MachinePostDominatorTree::createMachinePostDominatorTreePass() {
  return new MachinePostDominatorTree();
}

bool
MachinePostDominatorTree::runOnMachineFunction(MachineFunction &F) {
  DT->recalculate(F);
  return false;
}

MachinePostDominatorTree::~MachinePostDominatorTree() {
  delete DT;
}

void
MachinePostDominatorTree::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void
MachinePostDominatorTree::print(llvm::raw_ostream &OS, const Module *M) const {
  DT->print(OS);
}
