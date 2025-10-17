/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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

//===-- X86TargetObjectFile.cpp - X86 Object Info -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86TargetObjectFile.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ELF.h"
using namespace llvm;
using namespace dwarf;

const MCExpr *X86_64MachoTargetObjectFile::
getExprForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                               MachineModuleInfo *MMI, unsigned Encoding,
                               MCStreamer &Streamer) const {

  // On Darwin/X86-64, we can reference dwarf symbols with foo@GOTPCREL+4, which
  // is an indirect pc-relative reference.
  if (Encoding & (DW_EH_PE_indirect | DW_EH_PE_pcrel)) {
    const MCSymbol *Sym = Mang->getSymbol(GV);
    const MCExpr *Res =
      MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_GOTPCREL, getContext());
    const MCExpr *Four = MCConstantExpr::Create(4, getContext());
    return MCBinaryExpr::CreateAdd(Res, Four, getContext());
  }

  return TargetLoweringObjectFileMachO::
    getExprForDwarfGlobalReference(GV, Mang, MMI, Encoding, Streamer);
}

MCSymbol *X86_64MachoTargetObjectFile::
getCFIPersonalitySymbol(const GlobalValue *GV, Mangler *Mang,
                        MachineModuleInfo *MMI) const {
  return Mang->getSymbol(GV);
}

void
X86LinuxTargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);
}
