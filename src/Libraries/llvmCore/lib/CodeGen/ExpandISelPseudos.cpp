/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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

//===-- llvm/CodeGen/ExpandISelPseudos.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Expand Pseudo-instructions produced by ISel. These are usually to allow
// the expansion to contain control flow, such as a conditional move
// implemented with a conditional branch and a phi, or an atomic operation
// implemented with a loop.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "expand-isel-pseudos"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

namespace {
  class ExpandISelPseudos : public MachineFunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid
    ExpandISelPseudos() : MachineFunctionPass(ID) {}

  private:
    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
} // end anonymous namespace

char ExpandISelPseudos::ID = 0;
char &llvm::ExpandISelPseudosID = ExpandISelPseudos::ID;
INITIALIZE_PASS(ExpandISelPseudos, "expand-isel-pseudos",
                "Expand ISel Pseudo-instructions", false, false)

bool ExpandISelPseudos::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  const TargetLowering *TLI = MF.getTarget().getTargetLowering();

  // Iterate through each instruction in the function, looking for pseudos.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = I;
    for (MachineBasicBlock::iterator MBBI = MBB->begin(), MBBE = MBB->end();
         MBBI != MBBE; ) {
      MachineInstr *MI = MBBI++;

      // If MI is a pseudo, expand it.
      if (MI->usesCustomInsertionHook()) {
        Changed = true;
        MachineBasicBlock *NewMBB =
          TLI->EmitInstrWithCustomInserter(MI, MBB);
        // The expansion may involve new basic blocks.
        if (NewMBB != MBB) {
          MBB = NewMBB;
          I = NewMBB;
          MBBI = NewMBB->begin();
          MBBE = NewMBB->end();
        }
      }
    }
  }

  return Changed;
}
