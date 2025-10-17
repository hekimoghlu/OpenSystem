/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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

//===-- MSP430MCTargetDesc.cpp - MSP430 Target Descriptions ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides MSP430 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "MSP430MCTargetDesc.h"
#include "MSP430MCAsmInfo.h"
#include "InstPrinter/MSP430InstPrinter.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "MSP430GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "MSP430GenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "MSP430GenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createMSP430MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitMSP430MCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createMSP430MCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitMSP430MCRegisterInfo(X, MSP430::PCW);
  return X;
}

static MCSubtargetInfo *createMSP430MCSubtargetInfo(StringRef TT, StringRef CPU,
                                                    StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitMSP430MCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCCodeGenInfo *createMSP430MCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                                CodeModel::Model CM,
                                                CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCInstPrinter *createMSP430MCInstPrinter(const Target &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI,
                                                const MCSubtargetInfo &STI) {
  if (SyntaxVariant == 0)
    return new MSP430InstPrinter(MAI, MII, MRI);
  return 0;
}

extern "C" void LLVMInitializeMSP430TargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<MSP430MCAsmInfo> X(TheMSP430Target);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheMSP430Target,
                                        createMSP430MCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheMSP430Target, createMSP430MCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheMSP430Target,
                                    createMSP430MCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheMSP430Target,
                                          createMSP430MCSubtargetInfo);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheMSP430Target,
                                        createMSP430MCInstPrinter);
}
