/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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

//===-- XCoreRegisterInfo.h - XCore Register Information Impl ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the XCore implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef XCOREREGISTERINFO_H
#define XCOREREGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "XCoreGenRegisterInfo.inc"

namespace llvm {

class TargetInstrInfo;

struct XCoreRegisterInfo : public XCoreGenRegisterInfo {
private:
  const TargetInstrInfo &TII;

  void loadConstant(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  unsigned DstReg, int64_t Value, DebugLoc dl) const;

  void storeToStack(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  unsigned SrcReg, int Offset, DebugLoc dl) const;

  void loadFromStack(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  unsigned DstReg, int Offset, DebugLoc dl) const;

public:
  XCoreRegisterInfo(const TargetInstrInfo &tii);

  /// Code Generation virtual methods...

  const uint16_t *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;
  
  bool requiresRegisterScavenging(const MachineFunction &MF) const;

  bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const;

  bool useFPForScavengingIndex(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  // Debug information queries.
  unsigned getFrameRegister(const MachineFunction &MF) const;

  //! Return whether to emit frame moves
  static bool needsFrameMoves(const MachineFunction &MF);
};

} // end namespace llvm

#endif
