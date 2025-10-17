/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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

//===- NVPTXRegisterInfo.h - NVPTX Register Information Impl ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the NVPTX implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTXREGISTERINFO_H
#define NVPTXREGISTERINFO_H

#include "ManagedStringPool.h"
#include "llvm/Target/TargetRegisterInfo.h"


#define GET_REGINFO_HEADER
#include "NVPTXGenRegisterInfo.inc"
#include "llvm/Target/TargetRegisterInfo.h"
#include <sstream>

namespace llvm {

// Forward Declarations.
class TargetInstrInfo;
class NVPTXSubtarget;

class NVPTXRegisterInfo : public NVPTXGenRegisterInfo {
private:
  bool Is64Bit;
  // Hold Strings that can be free'd all together with NVPTXRegisterInfo
  ManagedStringPool     ManagedStrPool;

public:
  NVPTXRegisterInfo(const TargetInstrInfo &tii,
                    const NVPTXSubtarget &st);


  //------------------------------------------------------
  // Pure virtual functions from TargetRegisterInfo
  //------------------------------------------------------

  // NVPTX callee saved registers
  virtual const uint16_t*
  getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  // NVPTX callee saved register classes
  virtual const TargetRegisterClass* const *
  getCalleeSavedRegClasses(const MachineFunction *MF) const;

  virtual BitVector getReservedRegs(const MachineFunction &MF) const;

  virtual void eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                   int SPAdj,
                                   RegScavenger *RS=NULL) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  virtual int getDwarfRegNum(unsigned RegNum, bool isEH) const;
  virtual unsigned getFrameRegister(const MachineFunction &MF) const;
  virtual unsigned getRARegister() const;

  ManagedStringPool *getStrPool() const {
    return const_cast<ManagedStringPool *>(&ManagedStrPool);
  }

  const char *getName(unsigned RegNo) const {
    std::stringstream O;
    O << "reg" << RegNo;
    return getStrPool()->getManagedString(O.str().c_str())->c_str();
  }

};


std::string getNVPTXRegClassName (const TargetRegisterClass *RC);
std::string getNVPTXRegClassStr (const TargetRegisterClass *RC);
bool isNVPTXVectorRegClass (const TargetRegisterClass *RC);
std::string getNVPTXElemClassName (const TargetRegisterClass *RC);
int getNVPTXVectorSize (const TargetRegisterClass *RC);
const TargetRegisterClass *getNVPTXElemClass(const TargetRegisterClass *RC);

} // end namespace llvm


#endif
