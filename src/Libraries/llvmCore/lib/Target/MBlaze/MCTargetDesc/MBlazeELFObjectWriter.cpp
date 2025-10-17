/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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

//===-- MBlazeELFObjectWriter.cpp - MBlaze ELF Writer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MBlazeMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
  class MBlazeELFObjectWriter : public MCELFObjectTargetWriter {
  public:
    MBlazeELFObjectWriter(uint8_t OSABI);

    virtual ~MBlazeELFObjectWriter();
  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend) const;
  };
}

MBlazeELFObjectWriter::MBlazeELFObjectWriter(uint8_t OSABI)
  : MCELFObjectTargetWriter(/*Is64Bit*/ false, OSABI, ELF::EM_MBLAZE,
                            /*HasRelocationAddend*/ false) {}

MBlazeELFObjectWriter::~MBlazeELFObjectWriter() {
}

unsigned MBlazeELFObjectWriter::GetRelocType(const MCValue &Target,
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
    case FK_PCRel_4:
      Type = ELF::R_MICROBLAZE_64_PCREL;
      break;
    case FK_PCRel_2:
      Type = ELF::R_MICROBLAZE_32_PCREL;
      break;
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
    default: llvm_unreachable("invalid fixup kind!");
    case FK_Data_4:
      Type = ((IsRelocWithSymbol || Addend !=0)
              ? ELF::R_MICROBLAZE_32
              : ELF::R_MICROBLAZE_64);
      break;
    case FK_Data_2:
      Type = ELF::R_MICROBLAZE_32;
      break;
    }
  }
  return Type;
}



MCObjectWriter *llvm::createMBlazeELFObjectWriter(raw_ostream &OS,
                                                  uint8_t OSABI) {
  MCELFObjectTargetWriter *MOTW = new MBlazeELFObjectWriter(OSABI);
  return createELFObjectWriter(MOTW, OS,  /*IsLittleEndian=*/ false);
}
