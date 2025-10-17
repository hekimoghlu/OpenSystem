/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

//===-- MCELFObjectTargetWriter.cpp - ELF Target Writer Subclass ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCELFObjectWriter.h"

using namespace llvm;

MCELFObjectTargetWriter::MCELFObjectTargetWriter(bool Is64Bit_,
                                                 uint8_t OSABI_,
                                                 uint16_t EMachine_,
                                                 bool HasRelocationAddend_,
                                                 bool IsN64_)
  : OSABI(OSABI_), EMachine(EMachine_),
    HasRelocationAddend(HasRelocationAddend_), Is64Bit(Is64Bit_),
    IsN64(IsN64_){
}

/// Default e_flags = 0
unsigned MCELFObjectTargetWriter::getEFlags() const {
  return 0;
}

const MCSymbol *MCELFObjectTargetWriter::ExplicitRelSym(const MCAssembler &Asm,
                                                        const MCValue &Target,
                                                        const MCFragment &F,
                                                        const MCFixup &Fixup,
                                                        bool IsPCRel) const {
  return NULL;
}


void MCELFObjectTargetWriter::adjustFixupOffset(const MCFixup &Fixup,
                                                uint64_t &RelocOffset) {
}

void
MCELFObjectTargetWriter::sortRelocs(const MCAssembler &Asm,
                                    std::vector<ELFRelocationEntry> &Relocs) {
  // Sort by the r_offset, just like gnu as does.
  array_pod_sort(Relocs.begin(), Relocs.end());
}
