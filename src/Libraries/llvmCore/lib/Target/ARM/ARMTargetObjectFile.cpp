/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

//===-- llvm/Target/ARMTargetObjectFile.cpp - ARM Object Info Impl --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARMTargetObjectFile.h"
#include "ARMSubtarget.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ELF.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;
using namespace dwarf;

//===----------------------------------------------------------------------===//
//                               ELF Target
//===----------------------------------------------------------------------===//

void ARMElfTargetObjectFile::Initialize(MCContext &Ctx,
                                        const TargetMachine &TM) {
  bool isAAPCS_ABI = TM.getSubtarget<ARMSubtarget>().isAAPCS_ABI();
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(isAAPCS_ABI);

  if (isAAPCS_ABI) {
    LSDASection = NULL;
  }

  AttributesSection =
    getContext().getELFSection(".ARM.attributes",
                               ELF::SHT_ARM_ATTRIBUTES,
                               0,
                               SectionKind::getMetadata());
}
