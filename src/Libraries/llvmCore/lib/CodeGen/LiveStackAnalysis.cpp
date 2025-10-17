/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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

//===-- LiveStackAnalysis.cpp - Live Stack Slot Analysis ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the live stack slot analysis pass. It is analogous to
// live interval analysis except it's analyzing liveness of stack slots rather
// than registers.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "livestacks"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
#include <limits>
using namespace llvm;

char LiveStacks::ID = 0;
INITIALIZE_PASS_BEGIN(LiveStacks, "livestacks",
                "Live Stack Slot Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_END(LiveStacks, "livestacks",
                "Live Stack Slot Analysis", false, false)

char &llvm::LiveStacksID = LiveStacks::ID;

void LiveStacks::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addPreserved<SlotIndexes>();
  AU.addRequiredTransitive<SlotIndexes>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void LiveStacks::releaseMemory() {
  // Release VNInfo memory regions, VNInfo objects don't need to be dtor'd.
  VNInfoAllocator.Reset();
  S2IMap.clear();
  S2RCMap.clear();
}

bool LiveStacks::runOnMachineFunction(MachineFunction &MF) {
  TRI = MF.getTarget().getRegisterInfo();
  // FIXME: No analysis is being done right now. We are relying on the
  // register allocators to provide the information.
  return false;
}

LiveInterval &
LiveStacks::getOrCreateInterval(int Slot, const TargetRegisterClass *RC) {
  assert(Slot >= 0 && "Spill slot indice must be >= 0");
  SS2IntervalMap::iterator I = S2IMap.find(Slot);
  if (I == S2IMap.end()) {
    I = S2IMap.insert(I, std::make_pair(Slot,
            LiveInterval(TargetRegisterInfo::index2StackSlot(Slot), 0.0F)));
    S2RCMap.insert(std::make_pair(Slot, RC));
  } else {
    // Use the largest common subclass register class.
    const TargetRegisterClass *OldRC = S2RCMap[Slot];
    S2RCMap[Slot] = TRI->getCommonSubClass(OldRC, RC);
  }
  return I->second;
}

/// print - Implement the dump method.
void LiveStacks::print(raw_ostream &OS, const Module*) const {

  OS << "********** INTERVALS **********\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    I->second.print(OS);
    int Slot = I->first;
    const TargetRegisterClass *RC = getIntervalRegClass(Slot);
    if (RC)
      OS << " [" << RC->getName() << "]\n";
    else
      OS << " [Unknown]\n";
  }
}
