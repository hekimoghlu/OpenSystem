/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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

//===-- NVPTXTargetMachine.h - Define TargetMachine for NVPTX ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVPTX specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//


#ifndef NVPTX_TARGETMACHINE_H
#define NVPTX_TARGETMACHINE_H

#include "NVPTXInstrInfo.h"
#include "NVPTXISelLowering.h"
#include "NVPTXRegisterInfo.h"
#include "NVPTXSubtarget.h"
#include "NVPTXFrameLowering.h"
#include "ManagedStringPool.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

/// NVPTXTargetMachine
///
class NVPTXTargetMachine : public LLVMTargetMachine {
  NVPTXSubtarget        Subtarget;
  const TargetData      DataLayout;       // Calculates type size & alignment
  NVPTXInstrInfo        InstrInfo;
  NVPTXTargetLowering   TLInfo;
  TargetSelectionDAGInfo   TSInfo;

  // NVPTX does not have any call stack frame, but need a NVPTX specific
  // FrameLowering class because TargetFrameLowering is abstract.
  NVPTXFrameLowering       FrameLowering;

  // Hold Strings that can be free'd all together with NVPTXTargetMachine
  ManagedStringPool     ManagedStrPool;

  //bool addCommonCodeGenPasses(PassManagerBase &, CodeGenOpt::Level,
  //                            bool DisableVerify, MCContext *&OutCtx);

public:
  NVPTXTargetMachine(const Target &T, StringRef TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     Reloc::Model RM, CodeModel::Model CM,
                     CodeGenOpt::Level OP,
                     bool is64bit);

  virtual const TargetFrameLowering *getFrameLowering() const {
    return &FrameLowering;
  }
  virtual const NVPTXInstrInfo *getInstrInfo() const  { return &InstrInfo; }
  virtual const TargetData *getTargetData() const     { return &DataLayout;}
  virtual const NVPTXSubtarget *getSubtargetImpl() const { return &Subtarget;}

  virtual const NVPTXRegisterInfo *getRegisterInfo() const {
    return &(InstrInfo.getRegisterInfo());
  }

  virtual NVPTXTargetLowering *getTargetLowering() const {
    return const_cast<NVPTXTargetLowering*>(&TLInfo);
  }

  virtual const TargetSelectionDAGInfo *getSelectionDAGInfo() const {
    return &TSInfo;
  }

  //virtual bool addInstSelector(PassManagerBase &PM,
  //                             CodeGenOpt::Level OptLevel);

  //virtual bool addPreRegAlloc(PassManagerBase &, CodeGenOpt::Level);

  ManagedStringPool *getManagedStrPool() const {
    return const_cast<ManagedStringPool*>(&ManagedStrPool);
  }

  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);

  // Emission of machine code through JITCodeEmitter is not supported.
  virtual bool addPassesToEmitMachineCode(PassManagerBase &,
                                          JITCodeEmitter &,
                                          bool = true) {
    return true;
  }

  // Emission of machine code through MCJIT is not supported.
  virtual bool addPassesToEmitMC(PassManagerBase &,
                                 MCContext *&,
                                 raw_ostream &,
                                 bool = true) {
    return true;
  }

}; // NVPTXTargetMachine.

class NVPTXTargetMachine32 : public NVPTXTargetMachine {
  virtual void anchor();
public:
  NVPTXTargetMachine32(const Target &T, StringRef TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
};

class NVPTXTargetMachine64 : public NVPTXTargetMachine {
  virtual void anchor();
public:
  NVPTXTargetMachine64(const Target &T, StringRef TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);
};


} // end namespace llvm

#endif
