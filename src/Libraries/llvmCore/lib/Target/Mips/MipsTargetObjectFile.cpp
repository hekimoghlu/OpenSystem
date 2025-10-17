/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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

//===-- MipsTargetObjectFile.cpp - Mips Object Files ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetObjectFile.h"
#include "MipsSubtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ELF.h"
using namespace llvm;

static cl::opt<unsigned>
SSThreshold("mips-ssection-threshold", cl::Hidden,
            cl::desc("Small data and bss section threshold size (default=8)"),
            cl::init(8));

void MipsTargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &TM){
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);

  SmallDataSection =
    getContext().getELFSection(".sdata", ELF::SHT_PROGBITS,
                               ELF::SHF_WRITE |ELF::SHF_ALLOC,
                               SectionKind::getDataRel());

  SmallBSSSection =
    getContext().getELFSection(".sbss", ELF::SHT_NOBITS,
                               ELF::SHF_WRITE |ELF::SHF_ALLOC,
                               SectionKind::getBSS());

}

// A address must be loaded from a small section if its size is less than the
// small section size threshold. Data in this section must be addressed using
// gp_rel operator.
static bool IsInSmallSection(uint64_t Size) {
  return Size > 0 && Size <= SSThreshold;
}

bool MipsTargetObjectFile::IsGlobalInSmallSection(const GlobalValue *GV,
                                                const TargetMachine &TM) const {
  if (GV->isDeclaration() || GV->hasAvailableExternallyLinkage())
    return false;

  return IsGlobalInSmallSection(GV, TM, getKindForGlobal(GV, TM));
}

/// IsGlobalInSmallSection - Return true if this global address should be
/// placed into small data/bss section.
bool MipsTargetObjectFile::
IsGlobalInSmallSection(const GlobalValue *GV, const TargetMachine &TM,
                       SectionKind Kind) const {

  const MipsSubtarget &Subtarget = TM.getSubtarget<MipsSubtarget>();

  // Return if small section is not available.
  if (!Subtarget.useSmallSection())
    return false;

  // Only global variables, not functions.
  const GlobalVariable *GVA = dyn_cast<GlobalVariable>(GV);
  if (!GVA)
    return false;

  // We can only do this for datarel or BSS objects for now.
  if (!Kind.isBSS() && !Kind.isDataRel())
    return false;

  // If this is a internal constant string, there is a special
  // section for it, but not in small data/bss.
  if (Kind.isMergeable1ByteCString())
    return false;

  Type *Ty = GV->getType()->getElementType();
  return IsInSmallSection(TM.getTargetData()->getTypeAllocSize(Ty));
}



const MCSection *MipsTargetObjectFile::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {
  // TODO: Could also support "weak" symbols as well with ".gnu.linkonce.s.*"
  // sections?

  // Handle Small Section classification here.
  if (Kind.isBSS() && IsGlobalInSmallSection(GV, TM, Kind))
    return SmallBSSSection;
  if (Kind.isDataNoRel() && IsGlobalInSmallSection(GV, TM, Kind))
    return SmallDataSection;

  // Otherwise, we work the same as ELF.
  return TargetLoweringObjectFileELF::SelectSectionForGlobal(GV, Kind, Mang,TM);
}
