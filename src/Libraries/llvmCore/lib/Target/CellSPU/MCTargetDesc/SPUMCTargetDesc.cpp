/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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

//===-- SPUMCTargetDesc.cpp - Cell SPU Target Descriptions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Cell SPU specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "SPUMCTargetDesc.h"
#include "SPUMCAsmInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "SPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "SPUGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "SPUGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createSPUMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitSPUMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createCellSPUMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitSPUMCRegisterInfo(X, SPU::R0);
  return X;
}

static MCSubtargetInfo *createSPUMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                 StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitSPUMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCAsmInfo *createSPUMCAsmInfo(const Target &T, StringRef TT) {
  MCAsmInfo *MAI = new SPULinuxMCAsmInfo(T, TT);

  // Initial state of the frame pointer is R1.
  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(SPU::R1, 0);
  MAI->addInitialFrameState(0, Dst, Src);

  return MAI;
}

static MCCodeGenInfo *createSPUMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                             CodeModel::Model CM,
                                             CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  // For the time being, use static relocations, since there's really no
  // support for PIC yet.
  X->InitMCCodeGenInfo(Reloc::Static, CM, OL);
  return X;
}

// Force static initialization.
extern "C" void LLVMInitializeCellSPUTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheCellSPUTarget, createSPUMCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheCellSPUTarget,
                                        createSPUMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheCellSPUTarget, createSPUMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheCellSPUTarget,
                                    createCellSPUMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheCellSPUTarget,
                                          createSPUMCSubtargetInfo);
}
