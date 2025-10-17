/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

//===-- X86WinCOFFObjectWriter.cpp - X86 Win COFF Writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86FixupKinds.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/MC/MCWinCOFFObjectWriter.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace llvm {
  class MCObjectWriter;
}

namespace {
  class X86WinCOFFObjectWriter : public MCWinCOFFObjectTargetWriter {
    const bool Is64Bit;

  public:
    X86WinCOFFObjectWriter(bool Is64Bit_);
    ~X86WinCOFFObjectWriter();

    virtual unsigned getRelocType(unsigned FixupKind) const;
  };
}

X86WinCOFFObjectWriter::X86WinCOFFObjectWriter(bool Is64Bit_)
  : MCWinCOFFObjectTargetWriter(Is64Bit_ ? COFF::IMAGE_FILE_MACHINE_AMD64 :
                                COFF::IMAGE_FILE_MACHINE_I386),
    Is64Bit(Is64Bit_) {}

X86WinCOFFObjectWriter::~X86WinCOFFObjectWriter() {}

unsigned X86WinCOFFObjectWriter::getRelocType(unsigned FixupKind) const {
  switch (FixupKind) {
  case FK_PCRel_4:
  case X86::reloc_riprel_4byte:
  case X86::reloc_riprel_4byte_movq_load:
    return Is64Bit ? COFF::IMAGE_REL_AMD64_REL32 : COFF::IMAGE_REL_I386_REL32;
  case FK_Data_4:
  case X86::reloc_signed_4byte:
    return Is64Bit ? COFF::IMAGE_REL_AMD64_ADDR32 : COFF::IMAGE_REL_I386_DIR32;
  case FK_Data_8:
    if (Is64Bit)
      return COFF::IMAGE_REL_AMD64_ADDR64;
    llvm_unreachable("unsupported relocation type");
  case FK_SecRel_4:
    return Is64Bit ? COFF::IMAGE_REL_AMD64_SECREL : COFF::IMAGE_REL_I386_SECREL;
  default:
    llvm_unreachable("unsupported relocation type");
  }
}

MCObjectWriter *llvm::createX86WinCOFFObjectWriter(raw_ostream &OS,
                                                   bool Is64Bit) {
  MCWinCOFFObjectTargetWriter *MOTW = new X86WinCOFFObjectWriter(Is64Bit);
  return createWinCOFFObjectWriter(MOTW, OS);
}
