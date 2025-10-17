/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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

//===-- HexagonMCTargetDesc.cpp - Hexagon Target Descriptions -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Hexagon specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "HexagonMCTargetDesc.h"
#include "HexagonMCAsmInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "HexagonGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "HexagonGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "HexagonGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createHexagonMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitHexagonMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createHexagonMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitHexagonMCRegisterInfo(X, Hexagon::R0);
  return X;
}

static MCSubtargetInfo *createHexagonMCSubtargetInfo(StringRef TT,
                                                     StringRef CPU,
                                                     StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitHexagonMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCAsmInfo *createHexagonMCAsmInfo(const Target &T, StringRef TT) {
  MCAsmInfo *MAI = new HexagonMCAsmInfo(T, TT);

  // VirtualFP = (R30 + #0).
  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(Hexagon::R30, 0);
  MAI->addInitialFrameState(0, Dst, Src);

  return MAI;
}

static MCCodeGenInfo *createHexagonMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                             CodeModel::Model CM,
                                             CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  // For the time being, use static relocations, since there's really no
  // support for PIC yet.
  X->InitMCCodeGenInfo(Reloc::Static, CM, OL);
  return X;
}

// Force static initialization.
extern "C" void LLVMInitializeHexagonTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheHexagonTarget, createHexagonMCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheHexagonTarget,
                                        createHexagonMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheHexagonTarget, createHexagonMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheHexagonTarget,
                                    createHexagonMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheHexagonTarget,
                                          createHexagonMCSubtargetInfo);
}
