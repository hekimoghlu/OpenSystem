/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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

//===-- llvm/MC/MCInstrInfo.h - Target Instruction Info ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the target machine instruction set.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINSTRINFO_H
#define LLVM_MC_MCINSTRINFO_H

#include "llvm/MC/MCInstrDesc.h"
#include <cassert>

namespace llvm {

//---------------------------------------------------------------------------
///
/// MCInstrInfo - Interface to description of machine instruction set
///
class MCInstrInfo {
  const MCInstrDesc *Desc;          // Raw array to allow static init'n
  const unsigned *InstrNameIndices; // Array for name indices in InstrNameData
  const char *InstrNameData;        // Instruction name string pool
  unsigned NumOpcodes;              // Number of entries in the desc array

public:
  /// InitMCInstrInfo - Initialize MCInstrInfo, called by TableGen
  /// auto-generated routines. *DO NOT USE*.
  void InitMCInstrInfo(const MCInstrDesc *D, const unsigned *NI, const char *ND,
                       unsigned NO) {
    Desc = D;
    InstrNameIndices = NI;
    InstrNameData = ND;
    NumOpcodes = NO;
  }

  unsigned getNumOpcodes() const { return NumOpcodes; }

  /// get - Return the machine instruction descriptor that corresponds to the
  /// specified instruction opcode.
  ///
  const MCInstrDesc &get(unsigned Opcode) const {
    assert(Opcode < NumOpcodes && "Invalid opcode!");
    return Desc[Opcode];
  }

  /// getName - Returns the name for the instructions with the given opcode.
  const char *getName(unsigned Opcode) const {
    assert(Opcode < NumOpcodes && "Invalid opcode!");
    return &InstrNameData[InstrNameIndices[Opcode]];
  }
};

} // End llvm namespace

#endif
