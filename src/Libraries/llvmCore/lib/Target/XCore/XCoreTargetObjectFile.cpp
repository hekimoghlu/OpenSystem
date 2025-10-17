/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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

//===-- XCoreTargetObjectFile.cpp - XCore object files --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "XCoreTargetObjectFile.h"
#include "XCoreSubtarget.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/ELF.h"
using namespace llvm;


void XCoreTargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &TM){
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);

  DataSection =
    Ctx.getELFSection(".dp.data", ELF::SHT_PROGBITS, 
                      ELF::SHF_ALLOC | ELF::SHF_WRITE |
                      ELF::XCORE_SHF_DP_SECTION,
                      SectionKind::getDataRel());
  BSSSection =
    Ctx.getELFSection(".dp.bss", ELF::SHT_NOBITS,
                      ELF::SHF_ALLOC | ELF::SHF_WRITE |
                      ELF::XCORE_SHF_DP_SECTION,
                      SectionKind::getBSS());
  
  MergeableConst4Section = 
    Ctx.getELFSection(".cp.rodata.cst4", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC | ELF::SHF_MERGE |
                      ELF::XCORE_SHF_CP_SECTION,
                      SectionKind::getMergeableConst4());
  MergeableConst8Section = 
    Ctx.getELFSection(".cp.rodata.cst8", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC | ELF::SHF_MERGE |
                      ELF::XCORE_SHF_CP_SECTION,
                      SectionKind::getMergeableConst8());
  MergeableConst16Section = 
    Ctx.getELFSection(".cp.rodata.cst16", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC | ELF::SHF_MERGE |
                      ELF::XCORE_SHF_CP_SECTION,
                      SectionKind::getMergeableConst16());
  
  // TLS globals are lowered in the backend to arrays indexed by the current
  // thread id. After lowering they require no special handling by the linker
  // and can be placed in the standard data / bss sections.
  TLSDataSection = DataSection;
  TLSBSSSection = BSSSection;

  ReadOnlySection = 
    Ctx.getELFSection(".cp.rodata", ELF::SHT_PROGBITS,
                      ELF::SHF_ALLOC |
                      ELF::XCORE_SHF_CP_SECTION,
                      SectionKind::getReadOnlyWithRel());

  // Dynamic linking is not supported. Data with relocations is placed in the
  // same section as data without relocations.
  DataRelSection = DataRelLocalSection = DataSection;
  DataRelROSection = DataRelROLocalSection = ReadOnlySection;
}
