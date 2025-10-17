/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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

//===-- MBlazeMCTargetDesc.cpp - MBlaze Target Descriptions ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides MBlaze specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "MBlazeMCTargetDesc.h"
#include "MBlazeMCAsmInfo.h"
#include "InstPrinter/MBlazeInstPrinter.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "MBlazeGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "MBlazeGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "MBlazeGenRegisterInfo.inc"

using namespace llvm;


static MCInstrInfo *createMBlazeMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitMBlazeMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createMBlazeMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitMBlazeMCRegisterInfo(X, MBlaze::R15);
  return X;
}

static MCSubtargetInfo *createMBlazeMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                    StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitMBlazeMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCAsmInfo *createMCAsmInfo(const Target &T, StringRef TT) {
  Triple TheTriple(TT);
  switch (TheTriple.getOS()) {
  default:
    return new MBlazeMCAsmInfo();
  }
}

static MCCodeGenInfo *createMBlazeMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                                CodeModel::Model CM,
                                                CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  if (RM == Reloc::Default)
    RM = Reloc::Static;
  if (CM == CodeModel::Default)
    CM = CodeModel::Small;
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCStreamer *createMCStreamer(const Target &T, StringRef TT,
                                    MCContext &Ctx, MCAsmBackend &MAB,
                                    raw_ostream &_OS,
                                    MCCodeEmitter *_Emitter,
                                    bool RelaxAll,
                                    bool NoExecStack) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin()) {
    llvm_unreachable("MBlaze does not support Darwin MACH-O format");
  }

  if (TheTriple.isOSWindows()) {
    llvm_unreachable("MBlaze does not support Windows COFF format");
  }

  return createELFStreamer(Ctx, MAB, _OS, _Emitter, RelaxAll, NoExecStack);
}

static MCInstPrinter *createMBlazeMCInstPrinter(const Target &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI,
                                                const MCSubtargetInfo &STI) {
  if (SyntaxVariant == 0)
    return new MBlazeInstPrinter(MAI, MII, MRI);
  return 0;
}

// Force static initialization.
extern "C" void LLVMInitializeMBlazeTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheMBlazeTarget, createMCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheMBlazeTarget,
                                        createMBlazeMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheMBlazeTarget, createMBlazeMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheMBlazeTarget,
                                    createMBlazeMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheMBlazeTarget,
                                          createMBlazeMCSubtargetInfo);

  // Register the MC code emitter
  TargetRegistry::RegisterMCCodeEmitter(TheMBlazeTarget,
                                        llvm::createMBlazeMCCodeEmitter);

  // Register the asm backend
  TargetRegistry::RegisterMCAsmBackend(TheMBlazeTarget,
                                       createMBlazeAsmBackend);

  // Register the object streamer
  TargetRegistry::RegisterMCObjectStreamer(TheMBlazeTarget,
                                           createMCStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheMBlazeTarget,
                                        createMBlazeMCInstPrinter);
}
