/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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

//==- HexagonRegisterInfo.h - Hexagon Register Information Impl --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Hexagon implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#ifndef HexagonREGISTERINFO_H
#define HexagonREGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/MC/MachineLocation.h"

#define GET_REGINFO_HEADER
#include "HexagonGenRegisterInfo.inc"

//
//  We try not to hard code the reserved registers in our code,
//  so the following two macros were defined. However, there
//  are still a few places that R11 and R10 are hard wired.
//  See below. If, in the future, we decided to change the reserved
//  register. Don't forget changing the following places.
//
//  1. the "Defs" set of STriw_pred in HexagonInstrInfo.td
//  2. the "Defs" set of LDri_pred in HexagonInstrInfo.td
//  3. the definition of "IntRegs" in HexagonRegisterInfo.td
//  4. the definition of "DoubleRegs" in HexagonRegisterInfo.td
//
#define HEXAGON_RESERVED_REG_1 Hexagon::R10
#define HEXAGON_RESERVED_REG_2 Hexagon::R11

namespace llvm {

class HexagonSubtarget;
class HexagonInstrInfo;
class Type;

struct HexagonRegisterInfo : public HexagonGenRegisterInfo {
  HexagonSubtarget &Subtarget;
  const HexagonInstrInfo &TII;

  HexagonRegisterInfo(HexagonSubtarget &st, const HexagonInstrInfo &tii);

  /// Code Generation virtual methods...
  const uint16_t *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  const TargetRegisterClass* const* getCalleeSavedRegClasses(
                                     const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  /// determineFrameLayout - Determine the size of the frame and maximum call
  /// frame size.
  void determineFrameLayout(MachineFunction &MF) const;

  /// requiresRegisterScavenging - returns true since we may need scavenging for
  /// a temporary register when generating hardware loop instructions.
  bool requiresRegisterScavenging(const MachineFunction &MF) const {
    return true;
  }

  bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const {
    return true;
  }

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(const MachineFunction &MF) const;
  unsigned getFrameRegister() const;
  void getInitialFrameState(std::vector<MachineMove> &Moves) const;
  unsigned getStackRegister() const;

  // Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;
  const RegClassWeight &getRegClassWeight(const TargetRegisterClass *RC) const;
  unsigned getNumRegPressureSets() const;
  const char *getRegPressureSetName(unsigned Idx) const;
  unsigned getRegPressureSetLimit(unsigned Idx) const;
  const int* getRegClassPressureSets(const TargetRegisterClass *RC) const;
};

} // end namespace llvm

#endif
