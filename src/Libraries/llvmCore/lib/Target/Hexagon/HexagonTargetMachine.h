/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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

//=-- HexagonTargetMachine.h - Define TargetMachine for Hexagon ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Hexagon specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef HexagonTARGETMACHINE_H
#define HexagonTARGETMACHINE_H

#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonISelLowering.h"
#include "HexagonSelectionDAGInfo.h"
#include "HexagonFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

namespace llvm {

class Module;

class HexagonTargetMachine : public LLVMTargetMachine {
  const TargetData DataLayout;       // Calculates type size & alignment.
  HexagonSubtarget Subtarget;
  HexagonInstrInfo InstrInfo;
  HexagonTargetLowering TLInfo;
  HexagonSelectionDAGInfo TSInfo;
  HexagonFrameLowering FrameLowering;
  const InstrItineraryData* InstrItins;

public:
  HexagonTargetMachine(const Target &T, StringRef TT,StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);

  virtual const HexagonInstrInfo *getInstrInfo() const {
    return &InstrInfo;
  }
  virtual const HexagonSubtarget *getSubtargetImpl() const {
    return &Subtarget;
  }
  virtual const HexagonRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual const InstrItineraryData* getInstrItineraryData() const {
    return InstrItins;
  }


  virtual const HexagonTargetLowering* getTargetLowering() const {
    return &TLInfo;
  }

  virtual const HexagonFrameLowering* getFrameLowering() const {
    return &FrameLowering;
  }

  virtual const HexagonSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }

  virtual const TargetData       *getTargetData() const { return &DataLayout; }
  static unsigned getModuleMatchQuality(const Module &M);

  // Pass Pipeline Configuration.
  virtual bool addPassesForOptimizations(PassManagerBase &PM);
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
};

extern bool flag_aligned_memcpy;

} // end namespace llvm

#endif
