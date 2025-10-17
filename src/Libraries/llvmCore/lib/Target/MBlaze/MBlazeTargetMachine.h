/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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

//===-- MBlazeTargetMachine.h - Define TargetMachine for MBlaze -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MBlaze specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZE_TARGETMACHINE_H
#define MBLAZE_TARGETMACHINE_H

#include "MBlazeSubtarget.h"
#include "MBlazeInstrInfo.h"
#include "MBlazeISelLowering.h"
#include "MBlazeSelectionDAGInfo.h"
#include "MBlazeIntrinsicInfo.h"
#include "MBlazeFrameLowering.h"
#include "MBlazeELFWriterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class formatted_raw_ostream;

  class MBlazeTargetMachine : public LLVMTargetMachine {
    MBlazeSubtarget        Subtarget;
    const TargetData       DataLayout; // Calculates type size & alignment
    MBlazeInstrInfo        InstrInfo;
    MBlazeFrameLowering    FrameLowering;
    MBlazeTargetLowering   TLInfo;
    MBlazeSelectionDAGInfo TSInfo;
    MBlazeIntrinsicInfo    IntrinsicInfo;
    MBlazeELFWriterInfo    ELFWriterInfo;
    InstrItineraryData     InstrItins;

  public:
    MBlazeTargetMachine(const Target &T, StringRef TT,
                        StringRef CPU, StringRef FS,
                        const TargetOptions &Options,
                        Reloc::Model RM, CodeModel::Model CM,
                        CodeGenOpt::Level OL);

    virtual const MBlazeInstrInfo *getInstrInfo() const
    { return &InstrInfo; }

    virtual const InstrItineraryData *getInstrItineraryData() const
    {  return &InstrItins; }

    virtual const TargetFrameLowering *getFrameLowering() const
    { return &FrameLowering; }

    virtual const MBlazeSubtarget *getSubtargetImpl() const
    { return &Subtarget; }

    virtual const TargetData *getTargetData() const
    { return &DataLayout;}

    virtual const MBlazeRegisterInfo *getRegisterInfo() const
    { return &InstrInfo.getRegisterInfo(); }

    virtual const MBlazeTargetLowering *getTargetLowering() const
    { return &TLInfo; }

    virtual const MBlazeSelectionDAGInfo* getSelectionDAGInfo() const
    { return &TSInfo; }

    const TargetIntrinsicInfo *getIntrinsicInfo() const
    { return &IntrinsicInfo; }

    virtual const MBlazeELFWriterInfo *getELFWriterInfo() const {
      return &ELFWriterInfo;
    }

    // Pass Pipeline Configuration
    virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
  };
} // End llvm namespace

#endif
