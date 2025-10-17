/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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

//===-- PPCELFObjectWriter.cpp - PPC ELF Writer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PPCFixupKinds.h"
#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
  class PPCELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    PPCELFObjectWriter(bool Is64Bit, uint8_t OSABI);

    virtual ~PPCELFObjectWriter();
  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend) const;
    virtual void adjustFixupOffset(const MCFixup &Fixup, uint64_t &RelocOffset);
  };
}

PPCELFObjectWriter::PPCELFObjectWriter(bool Is64Bit, uint8_t OSABI)
  : MCELFObjectTargetWriter(Is64Bit, OSABI,
                            Is64Bit ?  ELF::EM_PPC64 : ELF::EM_PPC,
                            /*HasRelocationAddend*/ true) {}

PPCELFObjectWriter::~PPCELFObjectWriter() {
}

unsigned PPCELFObjectWriter::GetRelocType(const MCValue &Target,
                                             const MCFixup &Fixup,
                                             bool IsPCRel,
                                             bool IsRelocWithSymbol,
                                             int64_t Addend) const {
  // determine the type of the relocation
  unsigned Type;
  if (IsPCRel) {
    switch ((unsigned)Fixup.getKind()) {
    default:
      llvm_unreachable("Unimplemented");
    case PPC::fixup_ppc_br24:
      Type = ELF::R_PPC_REL24;
      break;
    case FK_PCRel_4:
      Type = ELF::R_PPC_REL32;
      break;
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");
    case PPC::fixup_ppc_br24:
      Type = ELF::R_PPC_ADDR24;
      break;
    case PPC::fixup_ppc_brcond14:
      Type = ELF::R_PPC_ADDR14_BRTAKEN; // XXX: or BRNTAKEN?_
      break;
    case PPC::fixup_ppc_ha16:
      Type = ELF::R_PPC_ADDR16_HA;
      break;
    case PPC::fixup_ppc_lo16:
      Type = ELF::R_PPC_ADDR16_LO;
      break;
    case PPC::fixup_ppc_lo14:
      Type = ELF::R_PPC_ADDR14;
      break;
    case FK_Data_4:
      Type = ELF::R_PPC_ADDR32;
      break;
    case FK_Data_2:
      Type = ELF::R_PPC_ADDR16;
      break;
    }
  }
  return Type;
}

void PPCELFObjectWriter::
adjustFixupOffset(const MCFixup &Fixup, uint64_t &RelocOffset) {
  switch ((unsigned)Fixup.getKind()) {
    case PPC::fixup_ppc_ha16:
    case PPC::fixup_ppc_lo16:
      RelocOffset += 2;
      break;
    default:
      break;
  }
}

MCObjectWriter *llvm::createPPCELFObjectWriter(raw_ostream &OS,
                                               bool Is64Bit,
                                               uint8_t OSABI) {
  MCELFObjectTargetWriter *MOTW = new PPCELFObjectWriter(Is64Bit, OSABI);
  return createELFObjectWriter(MOTW, OS,  /*IsLittleEndian=*/false);
}
