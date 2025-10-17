/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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

//===-- MipsInstrInfo.h - Mips Instruction Information ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSINSTRUCTIONINFO_H
#define MIPSINSTRUCTIONINFO_H

#include "Mips.h"
#include "MipsAnalyzeImmediate.h"
#include "MipsRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "MipsGenInstrInfo.inc"

namespace llvm {

class MipsInstrInfo : public MipsGenInstrInfo {
protected:
  MipsTargetMachine &TM;
  unsigned UncondBrOpc;

public:
  explicit MipsInstrInfo(MipsTargetMachine &TM, unsigned UncondBrOpc);

  static const MipsInstrInfo *create(MipsTargetMachine &TM);

  /// Branch Analysis
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify) const;

  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;

  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const SmallVectorImpl<MachineOperand> &Cond,
                                DebugLoc DL) const;

  virtual
  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;

  virtual MachineInstr* emitFrameIndexDebugValue(MachineFunction &MF,
                                                 int FrameIx, uint64_t Offset,
                                                 const MDNode *MDPtr,
                                                 DebugLoc DL) const;

  /// Insert nop instruction when hazard condition is found
  virtual void insertNoop(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI) const;

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MipsRegisterInfo &getRegisterInfo() const = 0;

  virtual unsigned GetOppositeBranchOpc(unsigned Opc) const = 0;

  /// Return the number of bytes of code the specified instruction may be.
  unsigned GetInstSizeInBytes(const MachineInstr *MI) const;

protected:
  bool isZeroImm(const MachineOperand &op) const;

  MachineMemOperand *GetMemOperand(MachineBasicBlock &MBB, int FI,
                                   unsigned Flag) const;

private:
  virtual unsigned GetAnalyzableBrOpc(unsigned Opc) const = 0;

  void AnalyzeCondBr(const MachineInstr *Inst, unsigned Opc,
                     MachineBasicBlock *&BB,
                     SmallVectorImpl<MachineOperand> &Cond) const;

  void BuildCondBr(MachineBasicBlock &MBB, MachineBasicBlock *TBB, DebugLoc DL,
                   const SmallVectorImpl<MachineOperand>& Cond) const;
};

/// Create MipsInstrInfo objects.
const MipsInstrInfo *createMips16InstrInfo(MipsTargetMachine &TM);
const MipsInstrInfo *createMipsSEInstrInfo(MipsTargetMachine &TM);

}

#endif
