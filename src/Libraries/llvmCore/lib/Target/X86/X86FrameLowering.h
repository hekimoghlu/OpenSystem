/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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

//===-- X86TargetFrameLowering.h - Define frame lowering for X86 -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements X86-specific bits of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#ifndef X86_FRAMELOWERING_H
#define X86_FRAMELOWERING_H

#include "X86Subtarget.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class MCSymbol;
  class X86TargetMachine;

class X86FrameLowering : public TargetFrameLowering {
  const X86TargetMachine &TM;
  const X86Subtarget &STI;
public:
  explicit X86FrameLowering(const X86TargetMachine &tm, const X86Subtarget &sti)
    : TargetFrameLowering(StackGrowsDown,
                          sti.getStackAlignment(),
                          (sti.is64Bit() ? -8 : -4)),
      TM(tm), STI(sti) {
  }

  void emitCalleeSavedFrameMoves(MachineFunction &MF, MCSymbol *Label,
                                 unsigned FramePtr) const;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  void adjustForSegmentedStacks(MachineFunction &MF) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS = NULL) const;

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

  int getFrameIndexOffset(const MachineFunction &MF, int FI) const;
  int getFrameIndexReference(const MachineFunction &MF, int FI,
                             unsigned &FrameReg) const;
  uint32_t getCompactUnwindEncoding(MachineFunction &MF) const;
};

} // End llvm namespace

#endif
