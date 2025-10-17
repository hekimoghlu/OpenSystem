/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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

//==-- ARMTargetFrameLowering.h - Define frame lowering for ARM --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef ARM_FRAMEINFO_H
#define ARM_FRAMEINFO_H

#include "ARM.h"
#include "ARMSubtarget.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class ARMSubtarget;

class ARMFrameLowering : public TargetFrameLowering {
protected:
  const ARMSubtarget &STI;

public:
  explicit ARMFrameLowering(const ARMSubtarget &sti)
    : TargetFrameLowering(StackGrowsDown, sti.getStackAlignment(), 0, 4),
      STI(sti) {
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI,
                                 const TargetRegisterInfo *TRI) const;

  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   const std::vector<CalleeSavedInfo> &CSI,
                                   const TargetRegisterInfo *TRI) const;

  bool hasFP(const MachineFunction &MF) const;
  bool hasReservedCallFrame(const MachineFunction &MF) const;
  bool canSimplifyCallFramePseudos(const MachineFunction &MF) const;
  int getFrameIndexReference(const MachineFunction &MF, int FI,
                             unsigned &FrameReg) const;
  int ResolveFrameIndexReference(const MachineFunction &MF,
                                 int FI,
                                 unsigned &FrameReg, int SPAdj) const;
  int getFrameIndexOffset(const MachineFunction &MF, int FI) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS) const;

 private:
  void emitPushInst(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                    const std::vector<CalleeSavedInfo> &CSI, unsigned StmOpc,
                    unsigned StrOpc, bool NoGap,
                    bool(*Func)(unsigned, bool), unsigned NumAlignedDPRCS2Regs,
                    unsigned MIFlags = 0) const;
  void emitPopInst(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   const std::vector<CalleeSavedInfo> &CSI, unsigned LdmOpc,
                   unsigned LdrOpc, bool isVarArg, bool NoGap,
                   bool(*Func)(unsigned, bool),
                   unsigned NumAlignedDPRCS2Regs) const;
};

} // End llvm namespace

#endif
