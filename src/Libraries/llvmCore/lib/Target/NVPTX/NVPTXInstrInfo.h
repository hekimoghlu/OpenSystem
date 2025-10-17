/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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

//===- NVPTXInstrInfo.h - NVPTX Instruction Information----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the niversity of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the NVPTX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTXINSTRUCTIONINFO_H
#define NVPTXINSTRUCTIONINFO_H

#include "NVPTX.h"
#include "NVPTXRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "NVPTXGenInstrInfo.inc"

namespace llvm {

class NVPTXInstrInfo : public NVPTXGenInstrInfo
{
  NVPTXTargetMachine &TM;
  const NVPTXRegisterInfo RegInfo;
public:
  explicit NVPTXInstrInfo(NVPTXTargetMachine &TM);

  virtual const NVPTXRegisterInfo &getRegisterInfo() const { return RegInfo; }

  /* The following virtual functions are used in register allocation.
   * They are not implemented because the existing interface and the logic
   * at the caller side do not work for the elementized vector load and store.
   *
   * virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
   *                                  int &FrameIndex) const;
   * virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
   *                                 int &FrameIndex) const;
   * virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
   *                              MachineBasicBlock::iterator MBBI,
   *                             unsigned SrcReg, bool isKill, int FrameIndex,
   *                              const TargetRegisterClass *RC) const;
   * virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
   *                               MachineBasicBlock::iterator MBBI,
   *                               unsigned DestReg, int FrameIndex,
   *                               const TargetRegisterClass *RC) const;
   */

  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const ;
  virtual bool isMoveInstr(const MachineInstr &MI,
                           unsigned &SrcReg,
                           unsigned &DestReg) const;
  bool isLoadInstr(const MachineInstr &MI, unsigned &AddrSpace) const;
  bool isStoreInstr(const MachineInstr &MI, unsigned &AddrSpace) const;
  bool isReadSpecialReg(MachineInstr &MI) const;

  virtual bool CanTailMerge(const MachineInstr *MI) const ;
  // Branch analysis.
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify) const;
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  virtual unsigned InsertBranch(MachineBasicBlock &MBB,MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const SmallVectorImpl<MachineOperand> &Cond,
                                DebugLoc DL) const;
  unsigned getLdStCodeAddrSpace(const MachineInstr &MI) const {
    return  MI.getOperand(2).getImm();
  }

};

} // namespace llvm

#endif
