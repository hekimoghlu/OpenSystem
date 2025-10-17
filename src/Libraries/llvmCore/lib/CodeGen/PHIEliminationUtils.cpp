/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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

//===-- PHIEliminationUtils.cpp - Helper functions for PHI elimination ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PHIEliminationUtils.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace llvm;

// findCopyInsertPoint - Find a safe place in MBB to insert a copy from SrcReg
// when following the CFG edge to SuccMBB. This needs to be after any def of
// SrcReg, but before any subsequent point where control flow might jump out of
// the basic block.
MachineBasicBlock::iterator
llvm::findPHICopyInsertPoint(MachineBasicBlock* MBB, MachineBasicBlock* SuccMBB,
                             unsigned SrcReg) {
  // Handle the trivial case trivially.
  if (MBB->empty())
    return MBB->begin();

  // Usually, we just want to insert the copy before the first terminator
  // instruction. However, for the edge going to a landing pad, we must insert
  // the copy before the call/invoke instruction.
  if (!SuccMBB->isLandingPad())
    return MBB->getFirstTerminator();

  // Discover any defs/uses in this basic block.
  SmallPtrSet<MachineInstr*, 8> DefUsesInMBB;
  MachineRegisterInfo& MRI = MBB->getParent()->getRegInfo();
  for (MachineRegisterInfo::reg_iterator RI = MRI.reg_begin(SrcReg),
         RE = MRI.reg_end(); RI != RE; ++RI) {
    MachineInstr* DefUseMI = &*RI;
    if (DefUseMI->getParent() == MBB)
      DefUsesInMBB.insert(DefUseMI);
  }

  MachineBasicBlock::iterator InsertPoint;
  if (DefUsesInMBB.empty()) {
    // No defs.  Insert the copy at the start of the basic block.
    InsertPoint = MBB->begin();
  } else if (DefUsesInMBB.size() == 1) {
    // Insert the copy immediately after the def/use.
    InsertPoint = *DefUsesInMBB.begin();
    ++InsertPoint;
  } else {
    // Insert the copy immediately after the last def/use.
    InsertPoint = MBB->end();
    while (!DefUsesInMBB.count(&*--InsertPoint)) {}
    ++InsertPoint;
  }

  // Make sure the copy goes after any phi nodes however.
  return MBB->SkipPHIsAndLabels(InsertPoint);
}
