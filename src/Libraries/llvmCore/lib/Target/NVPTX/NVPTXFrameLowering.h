/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 17, 2022.
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

//===--- NVPTXFrameLowering.h - Define frame lowering for NVPTX -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef NVPTX_FRAMELOWERING_H
#define NVPTX_FRAMELOWERING_H

#include "llvm/Target/TargetFrameLowering.h"


namespace llvm {
class NVPTXTargetMachine;

class NVPTXFrameLowering : public TargetFrameLowering {
  NVPTXTargetMachine &tm;
  bool is64bit;

public:
  explicit NVPTXFrameLowering(NVPTXTargetMachine &_tm, bool _is64bit)
  : TargetFrameLowering(TargetFrameLowering::StackGrowsUp, 8, 0),
    tm(_tm), is64bit(_is64bit) {}

  virtual bool hasFP(const MachineFunction &MF) const;
  virtual void emitPrologue(MachineFunction &MF) const;
  virtual void emitEpilogue(MachineFunction &MF,
                            MachineBasicBlock &MBB) const;
};

} // End llvm namespace

#endif
