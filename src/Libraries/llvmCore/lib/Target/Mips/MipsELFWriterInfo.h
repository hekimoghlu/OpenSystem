/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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

//===-- MipsELFWriterInfo.h - ELF Writer Info for Mips ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF writer information for the Mips backend.
//
//===----------------------------------------------------------------------===//

#ifndef MIPS_ELF_WRITER_INFO_H
#define MIPS_ELF_WRITER_INFO_H

#include "llvm/Target/TargetELFWriterInfo.h"

namespace llvm {

  class MipsELFWriterInfo : public TargetELFWriterInfo {

  public:
    MipsELFWriterInfo(bool is64Bit_, bool isLittleEndian_);
    virtual ~MipsELFWriterInfo();

    /// getRelocationType - Returns the target specific ELF Relocation type.
    /// 'MachineRelTy' contains the object code independent relocation type
    virtual unsigned getRelocationType(unsigned MachineRelTy) const;

    /// hasRelocationAddend - True if the target uses an addend in the
    /// ELF relocation entry.
    virtual bool hasRelocationAddend() const { return is64Bit ? true : false; }

    /// getDefaultAddendForRelTy - Gets the default addend value for a
    /// relocation entry based on the target ELF relocation type.
    virtual long int getDefaultAddendForRelTy(unsigned RelTy,
                                              long int Modifier = 0) const;

    /// getRelTySize - Returns the size of relocatable field in bits
    virtual unsigned getRelocationTySize(unsigned RelTy) const;

    /// isPCRelativeRel - True if the relocation type is pc relative
    virtual bool isPCRelativeRel(unsigned RelTy) const;

    /// getJumpTableRelocationTy - Returns the machine relocation type used
    /// to reference a jumptable.
    virtual unsigned getAbsoluteLabelMachineRelTy() const;

    /// computeRelocation - Some relocatable fields could be relocated
    /// directly, avoiding the relocation symbol emission, compute the
    /// final relocation value for this symbol.
    virtual long int computeRelocation(unsigned SymOffset, unsigned RelOffset,
                                       unsigned RelTy) const;
  };

} // end llvm namespace

#endif // MIPS_ELF_WRITER_INFO_H
