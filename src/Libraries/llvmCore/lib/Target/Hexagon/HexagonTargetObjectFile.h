/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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

//===-- HexagonTargetAsmInfo.h - Hexagon asm properties --------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HexagonTARGETOBJECTFILE_H
#define HexagonTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/MCSectionELF.h"

namespace llvm {

  class HexagonTargetObjectFile : public TargetLoweringObjectFileELF {
    const MCSectionELF *SmallDataSection;
    const MCSectionELF *SmallBSSSection;
  public:
    virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);

    /// IsGlobalInSmallSection - Return true if this global address should be
    /// placed into small data/bss section.
    bool IsGlobalInSmallSection(const GlobalValue *GV,
                                const TargetMachine &TM,
                                SectionKind Kind) const;
    bool IsGlobalInSmallSection(const GlobalValue *GV,
                                const TargetMachine &TM) const;

    const MCSection* SelectSectionForGlobal(const GlobalValue *GV,
                                            SectionKind Kind,
                                            Mangler *Mang,
                                            const TargetMachine &TM) const;
  };

} // namespace llvm

#endif
