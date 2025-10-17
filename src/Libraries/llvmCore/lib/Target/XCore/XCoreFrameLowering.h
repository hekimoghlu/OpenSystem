/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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

//===-- XCoreFrameLowering.h - Frame info for XCore Target ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains XCore frame information that doesn't fit anywhere else
// cleanly...
//
//===----------------------------------------------------------------------===//

#ifndef XCOREFRAMEINFO_H
#define XCOREFRAMEINFO_H

#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class XCoreSubtarget;

  class XCoreFrameLowering: public TargetFrameLowering {
  public:
    XCoreFrameLowering(const XCoreSubtarget &STI);

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

    void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                              RegScavenger *RS = NULL) const;

    //! Stack slot size (4 bytes)
    static int stackSlotSize() {
      return 4;
    }
  };
}

#endif // XCOREFRAMEINFO_H
