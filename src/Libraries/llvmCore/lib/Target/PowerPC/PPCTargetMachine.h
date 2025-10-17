/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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

//===-- PPCTargetMachine.h - Define TargetMachine for PowerPC ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef PPC_TARGETMACHINE_H
#define PPC_TARGETMACHINE_H

#include "PPCFrameLowering.h"
#include "PPCSubtarget.h"
#include "PPCJITInfo.h"
#include "PPCInstrInfo.h"
#include "PPCISelLowering.h"
#include "PPCSelectionDAGInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

namespace llvm {

/// PPCTargetMachine - Common code between 32-bit and 64-bit PowerPC targets.
///
class PPCTargetMachine : public LLVMTargetMachine {
  PPCSubtarget        Subtarget;
  const TargetData    DataLayout;       // Calculates type size & alignment
  PPCInstrInfo        InstrInfo;
  PPCFrameLowering    FrameLowering;
  PPCJITInfo          JITInfo;
  PPCTargetLowering   TLInfo;
  PPCSelectionDAGInfo TSInfo;
  InstrItineraryData  InstrItins;

public:
  PPCTargetMachine(const Target &T, StringRef TT,
                   StringRef CPU, StringRef FS, const TargetOptions &Options,
                   Reloc::Model RM, CodeModel::Model CM,
                   CodeGenOpt::Level OL, bool is64Bit);

  virtual const PPCInstrInfo      *getInstrInfo() const { return &InstrInfo; }
  virtual const PPCFrameLowering  *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual       PPCJITInfo        *getJITInfo()         { return &JITInfo; }
  virtual const PPCTargetLowering *getTargetLowering() const {
   return &TLInfo;
  }
  virtual const PPCSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }
  virtual const PPCRegisterInfo   *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual const TargetData    *getTargetData() const    { return &DataLayout; }
  virtual const PPCSubtarget  *getSubtargetImpl() const { return &Subtarget; }
  virtual const InstrItineraryData *getInstrItineraryData() const {
    return &InstrItins;
  }

  // Pass Pipeline Configuration
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
  virtual bool addCodeEmitter(PassManagerBase &PM,
                              JITCodeEmitter &JCE);
};

/// PPC32TargetMachine - PowerPC 32-bit target machine.
///
class PPC32TargetMachine : public PPCTargetMachine {
  virtual void anchor();
public:
  PPC32TargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
};

/// PPC64TargetMachine - PowerPC 64-bit target machine.
///
class PPC64TargetMachine : public PPCTargetMachine {
  virtual void anchor();
public:
  PPC64TargetMachine(const Target &T, StringRef TT,
                     StringRef CPU, StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
};

} // end namespace llvm

#endif
