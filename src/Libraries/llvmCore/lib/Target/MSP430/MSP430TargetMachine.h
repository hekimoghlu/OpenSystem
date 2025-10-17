/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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

//===-- MSP430TargetMachine.h - Define TargetMachine for MSP430 -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MSP430 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_TARGET_MSP430_TARGETMACHINE_H
#define LLVM_TARGET_MSP430_TARGETMACHINE_H

#include "MSP430InstrInfo.h"
#include "MSP430ISelLowering.h"
#include "MSP430FrameLowering.h"
#include "MSP430SelectionDAGInfo.h"
#include "MSP430RegisterInfo.h"
#include "MSP430Subtarget.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// MSP430TargetMachine
///
class MSP430TargetMachine : public LLVMTargetMachine {
  MSP430Subtarget        Subtarget;
  const TargetData       DataLayout;       // Calculates type size & alignment
  MSP430InstrInfo        InstrInfo;
  MSP430TargetLowering   TLInfo;
  MSP430SelectionDAGInfo TSInfo;
  MSP430FrameLowering    FrameLowering;

public:
  MSP430TargetMachine(const Target &T, StringRef TT,
                      StringRef CPU, StringRef FS, const TargetOptions &Options,
                      Reloc::Model RM, CodeModel::Model CM,
                      CodeGenOpt::Level OL);

  virtual const TargetFrameLowering *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual const MSP430InstrInfo *getInstrInfo() const  { return &InstrInfo; }
  virtual const TargetData *getTargetData() const     { return &DataLayout;}
  virtual const MSP430Subtarget *getSubtargetImpl() const { return &Subtarget; }

  virtual const TargetRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual const MSP430TargetLowering *getTargetLowering() const {
    return &TLInfo;
  }

  virtual const MSP430SelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }

  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
}; // MSP430TargetMachine.

} // end namespace llvm

#endif // LLVM_TARGET_MSP430_TARGETMACHINE_H
