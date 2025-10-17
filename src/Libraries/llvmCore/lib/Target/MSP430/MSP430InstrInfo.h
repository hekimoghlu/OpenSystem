/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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

//===-- MSP430InstrInfo.h - MSP430 Instruction Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSP430 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MSP430INSTRINFO_H
#define LLVM_TARGET_MSP430INSTRINFO_H

#include "MSP430RegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "MSP430GenInstrInfo.inc"

namespace llvm {

class MSP430TargetMachine;

/// MSP430II - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace MSP430II {
  enum {
    SizeShift   = 2,
    SizeMask    = 7 << SizeShift,

    SizeUnknown = 0 << SizeShift,
    SizeSpecial = 1 << SizeShift,
    Size2Bytes  = 2 << SizeShift,
    Size4Bytes  = 3 << SizeShift,
    Size6Bytes  = 4 << SizeShift
  };
}

class MSP430InstrInfo : public MSP430GenInstrInfo {
  const MSP430RegisterInfo RI;
public:
  explicit MSP430InstrInfo(MSP430TargetMachine &TM);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const TargetRegisterInfo &getRegisterInfo() const { return RI; }

  void copyPhysReg(MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator I, DebugLoc DL,
                   unsigned DestReg, unsigned SrcReg,
                   bool KillSrc) const;

  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned SrcReg, bool isKill,
                                   int FrameIndex,
                                   const TargetRegisterClass *RC,
                                   const TargetRegisterInfo *TRI) const;
  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned DestReg, int FrameIdx,
                                    const TargetRegisterClass *RC,
                                    const TargetRegisterInfo *TRI) const;

  unsigned GetInstSizeInBytes(const MachineInstr *MI) const;

  // Branch folding goodness
  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;
  bool isUnpredicatedTerminator(const MachineInstr *MI) const;
  bool AnalyzeBranch(MachineBasicBlock &MBB,
                     MachineBasicBlock *&TBB, MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify) const;

  unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB,
                        const SmallVectorImpl<MachineOperand> &Cond,
                        DebugLoc DL) const;

};

}

#endif
