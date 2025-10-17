/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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

//===-- MipsRegisterInfo.h - Mips Register Information Impl -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSREGISTERINFO_H
#define MIPSREGISTERINFO_H

#include "Mips.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "MipsGenRegisterInfo.inc"

namespace llvm {
class MipsSubtarget;
class Type;

class MipsRegisterInfo : public MipsGenRegisterInfo {
protected:
  const MipsSubtarget &Subtarget;

public:
  MipsRegisterInfo(const MipsSubtarget &Subtarget);

  /// getRegisterNumbering - Given the enum value for some register, e.g.
  /// Mips::RA, return the number that it corresponds to (e.g. 31).
  static unsigned getRegisterNumbering(unsigned RegEnum);

  /// Get PIC indirect call register
  static unsigned getPICCallReg();

  /// Adjust the Mips stack frame.
  void adjustMipsStackFrame(MachineFunction &MF) const;

  /// Code Generation virtual methods...
  const uint16_t *getCalleeSavedRegs(const MachineFunction *MF = 0) const;
  const uint32_t *getCallPreservedMask(CallingConv::ID) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  virtual bool requiresRegisterScavenging(const MachineFunction &MF) const;

  virtual bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const;

  /// Stack Frame Processing Methods
  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

  /// Debug information queries.
  unsigned getFrameRegister(const MachineFunction &MF) const;

  /// Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;

private:
  virtual void eliminateFI(MachineBasicBlock::iterator II, unsigned OpNo,
                           int FrameIndex, uint64_t StackSize,
                           int64_t SPOffset) const = 0;
};

} // end namespace llvm

#endif
