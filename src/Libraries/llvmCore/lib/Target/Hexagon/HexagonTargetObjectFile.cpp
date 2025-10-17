/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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

//===-- HexagonTargetObjectFile.cpp - Hexagon asm properties --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the HexagonTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "HexagonTargetObjectFile.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Target/TargetData.h"
#include "llvm/DerivedTypes.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<int> SmallDataThreshold("hexagon-small-data-threshold",
                                cl::init(8), cl::Hidden);

void HexagonTargetObjectFile::Initialize(MCContext &Ctx,
                                         const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);


  SmallDataSection =
    getContext().getELFSection(".sdata", ELF::SHT_PROGBITS,
                               ELF::SHF_WRITE | ELF::SHF_ALLOC,
                               SectionKind::getDataRel());
  SmallBSSSection =
    getContext().getELFSection(".sbss", ELF::SHT_NOBITS,
                               ELF::SHF_WRITE | ELF::SHF_ALLOC,
                               SectionKind::getBSS());
}

// sdata/sbss support taken largely from the MIPS Backend.
static bool IsInSmallSection(uint64_t Size) {
  return Size > 0 && Size <= (uint64_t)SmallDataThreshold;
}
/// IsGlobalInSmallSection - Return true if this global value should be
/// placed into small data/bss section.
bool HexagonTargetObjectFile::IsGlobalInSmallSection(const GlobalValue *GV,
                                                const TargetMachine &TM) const {
  // If the primary definition of this global value is outside the current
  // translation unit or the global value is available for inspection but not
  // emission, then do nothing.
  if (GV->isDeclaration() || GV->hasAvailableExternallyLinkage())
    return false;

  // Otherwise, Check if GV should be in sdata/sbss, when normally it would end
  // up in getKindForGlobal(GV, TM).
  return IsGlobalInSmallSection(GV, TM, getKindForGlobal(GV, TM));
}

/// IsGlobalInSmallSection - Return true if this global value should be
/// placed into small data/bss section.
bool HexagonTargetObjectFile::
IsGlobalInSmallSection(const GlobalValue *GV, const TargetMachine &TM,
                       SectionKind Kind) const {
  // Only global variables, not functions.
  const GlobalVariable *GVA = dyn_cast<GlobalVariable>(GV);
  if (!GVA)
    return false;

  if (Kind.isBSS() || Kind.isDataNoRel() || Kind.isCommon()) {
    Type *Ty = GV->getType()->getElementType();
    return IsInSmallSection(TM.getTargetData()->getTypeAllocSize(Ty));
  }

  return false;
}

const MCSection *HexagonTargetObjectFile::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {

  // Handle Small Section classification here.
  if (Kind.isBSS() && IsGlobalInSmallSection(GV, TM, Kind))
    return SmallBSSSection;
  if (Kind.isDataNoRel() && IsGlobalInSmallSection(GV, TM, Kind))
    return SmallDataSection;

  // Otherwise, we work the same as ELF.
  return TargetLoweringObjectFileELF::SelectSectionForGlobal(GV, Kind, Mang,TM);
}
