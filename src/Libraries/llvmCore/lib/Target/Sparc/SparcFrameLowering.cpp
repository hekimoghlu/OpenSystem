/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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

//===-- SparcFrameLowering.cpp - Sparc Frame Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Sparc implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "SparcFrameLowering.h"
#include "SparcInstrInfo.h"
#include "SparcMachineFunctionInfo.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

void SparcFrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const SparcInstrInfo &TII =
    *static_cast<const SparcInstrInfo*>(MF.getTarget().getInstrInfo());
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Get the number of bytes to allocate from the FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  // Emit the correct save instruction based on the number of bytes in
  // the frame. Minimum stack frame size according to V8 ABI is:
  //   16 words for register window spill
  //    1 word for address of returned aggregate-value
  // +  6 words for passing parameters on the stack
  // ----------
  //   23 words * 4 bytes per word = 92 bytes
  NumBytes += 92;

  // Round up to next doubleword boundary -- a double-word boundary
  // is required by the ABI.
  NumBytes = (NumBytes + 7) & ~7;
  NumBytes = -NumBytes;

  if (NumBytes >= -4096) {
    BuildMI(MBB, MBBI, dl, TII.get(SP::SAVEri), SP::O6)
      .addReg(SP::O6).addImm(NumBytes);
  } else {
    // Emit this the hard way.  This clobbers G1 which we always know is
    // available here.
    unsigned OffHi = (unsigned)NumBytes >> 10U;
    BuildMI(MBB, MBBI, dl, TII.get(SP::SETHIi), SP::G1).addImm(OffHi);
    // Emit G1 = G1 + I6
    BuildMI(MBB, MBBI, dl, TII.get(SP::ORri), SP::G1)
      .addReg(SP::G1).addImm(NumBytes & ((1 << 10)-1));
    BuildMI(MBB, MBBI, dl, TII.get(SP::SAVErr), SP::O6)
      .addReg(SP::O6).addReg(SP::G1);
  }
}

void SparcFrameLowering::emitEpilogue(MachineFunction &MF,
                                  MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  const SparcInstrInfo &TII =
    *static_cast<const SparcInstrInfo*>(MF.getTarget().getInstrInfo());
  DebugLoc dl = MBBI->getDebugLoc();
  assert(MBBI->getOpcode() == SP::RETL &&
         "Can only put epilog before 'retl' instruction!");
  BuildMI(MBB, MBBI, dl, TII.get(SP::RESTORErr), SP::G0).addReg(SP::G0)
    .addReg(SP::G0);
}
