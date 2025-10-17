/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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

//===- Thumb2RegisterInfo.h - Thumb-2 Register Information Impl -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-2 implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#ifndef THUMB2REGISTERINFO_H
#define THUMB2REGISTERINFO_H

#include "ARM.h"
#include "ARMBaseRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class ARMSubtarget;
  class ARMBaseInstrInfo;

struct Thumb2RegisterInfo : public ARMBaseRegisterInfo {
public:
  Thumb2RegisterInfo(const ARMBaseInstrInfo &tii, const ARMSubtarget &STI);

  /// emitLoadConstPool - Emits a load from constpool to materialize the
  /// specified immediate.
  void emitLoadConstPool(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator &MBBI,
                         DebugLoc dl,
                         unsigned DestReg, unsigned SubIdx, int Val,
                         ARMCC::CondCodes Pred = ARMCC::AL,
                         unsigned PredReg = 0,
                         unsigned MIFlags = MachineInstr::NoFlags) const;
};
}

#endif // THUMB2REGISTERINFO_H
