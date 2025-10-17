/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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

//===-- NVPTXMCTargetDesc.cpp - NVPTX Target Descriptions -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides NVPTX specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "NVPTXMCTargetDesc.h"
#include "NVPTXMCAsmInfo.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "NVPTXGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "NVPTXGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "NVPTXGenRegisterInfo.inc"


using namespace llvm;

static MCInstrInfo *createNVPTXMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitNVPTXMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createNVPTXMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  // PTX does not have a return address register.
  InitNVPTXMCRegisterInfo(X, 0);
  return X;
}

static MCSubtargetInfo *createNVPTXMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                   StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitNVPTXMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCCodeGenInfo *createNVPTXMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                               CodeModel::Model CM,
                                               CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}


// Force static initialization.
extern "C" void LLVMInitializeNVPTXTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<NVPTXMCAsmInfo> X(TheNVPTXTarget32);
  RegisterMCAsmInfo<NVPTXMCAsmInfo> Y(TheNVPTXTarget64);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheNVPTXTarget32,
                                        createNVPTXMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheNVPTXTarget64,
                                        createNVPTXMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheNVPTXTarget32, createNVPTXMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheNVPTXTarget64, createNVPTXMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheNVPTXTarget32,
                                    createNVPTXMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheNVPTXTarget64,
                                    createNVPTXMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheNVPTXTarget32,
                                          createNVPTXMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheNVPTXTarget64,
                                          createNVPTXMCSubtargetInfo);

}
