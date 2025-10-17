/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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

//===- MachineDominators.cpp - Machine Dominator Calculation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements simple dominator construction algorithms for finding
// forward dominators on machine functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/Passes.h"

using namespace llvm;

namespace llvm {
TEMPLATE_INSTANTIATION(class DomTreeNodeBase<MachineBasicBlock>);
TEMPLATE_INSTANTIATION(class DominatorTreeBase<MachineBasicBlock>);
}

char MachineDominatorTree::ID = 0;

INITIALIZE_PASS(MachineDominatorTree, "machinedomtree",
                "MachineDominator Tree Construction", true, true)

char &llvm::MachineDominatorsID = MachineDominatorTree::ID;

void MachineDominatorTree::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineDominatorTree::runOnMachineFunction(MachineFunction &F) {
  DT->recalculate(F);

  return false;
}

MachineDominatorTree::MachineDominatorTree()
    : MachineFunctionPass(ID) {
  initializeMachineDominatorTreePass(*PassRegistry::getPassRegistry());
  DT = new DominatorTreeBase<MachineBasicBlock>(false);
}

MachineDominatorTree::~MachineDominatorTree() {
  delete DT;
}

void MachineDominatorTree::releaseMemory() {
  DT->releaseMemory();
}

void MachineDominatorTree::print(raw_ostream &OS, const Module*) const {
  DT->print(OS);
}
