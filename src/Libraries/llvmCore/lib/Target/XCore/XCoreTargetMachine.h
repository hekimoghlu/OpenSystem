/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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

//===-- XCoreTargetMachine.h - Define TargetMachine for XCore ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the XCore specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef XCORETARGETMACHINE_H
#define XCORETARGETMACHINE_H

#include "XCoreFrameLowering.h"
#include "XCoreSubtarget.h"
#include "XCoreInstrInfo.h"
#include "XCoreISelLowering.h"
#include "XCoreSelectionDAGInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

namespace llvm {

class XCoreTargetMachine : public LLVMTargetMachine {
  XCoreSubtarget Subtarget;
  const TargetData DataLayout;       // Calculates type size & alignment
  XCoreInstrInfo InstrInfo;
  XCoreFrameLowering FrameLowering;
  XCoreTargetLowering TLInfo;
  XCoreSelectionDAGInfo TSInfo;
public:
  XCoreTargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);

  virtual const XCoreInstrInfo *getInstrInfo() const { return &InstrInfo; }
  virtual const XCoreFrameLowering *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual const XCoreSubtarget *getSubtargetImpl() const { return &Subtarget; }
  virtual const XCoreTargetLowering *getTargetLowering() const {
    return &TLInfo;
  }

  virtual const XCoreSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }

  virtual const TargetRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const TargetData       *getTargetData() const { return &DataLayout; }

  // Pass Pipeline Configuration
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
};

} // end namespace llvm

#endif
