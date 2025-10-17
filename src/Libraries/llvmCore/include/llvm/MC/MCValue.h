/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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

//===-- llvm/MC/MCValue.h - MCValue class -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCVALUE_H
#define LLVM_MC_MCVALUE_H

#include "llvm/Support/DataTypes.h"
#include "llvm/MC/MCSymbol.h"
#include <cassert>

namespace llvm {
class MCAsmInfo;
class MCSymbol;
class MCSymbolRefExpr;
class raw_ostream;

/// MCValue - This represents an "assembler immediate".  In its most general
/// form, this can hold "SymbolA - SymbolB + imm64".  Not all targets supports
/// relocations of this general form, but we need to represent this anyway.
///
/// In the general form, SymbolB can only be defined if SymbolA is, and both
/// must be in the same (non-external) section. The latter constraint is not
/// enforced, since a symbol's section may not be known at construction.
///
/// Note that this class must remain a simple POD value class, because we need
/// it to live in unions etc.
class MCValue {
  const MCSymbolRefExpr *SymA, *SymB;
  int64_t Cst;
public:

  int64_t getConstant() const { return Cst; }
  const MCSymbolRefExpr *getSymA() const { return SymA; }
  const MCSymbolRefExpr *getSymB() const { return SymB; }

  /// isAbsolute - Is this an absolute (as opposed to relocatable) value.
  bool isAbsolute() const { return !SymA && !SymB; }

  /// print - Print the value to the stream \p OS.
  void print(raw_ostream &OS, const MCAsmInfo *MAI) const;

  /// dump - Print the value to stderr.
  void dump() const;

  static MCValue get(const MCSymbolRefExpr *SymA, const MCSymbolRefExpr *SymB=0,
                     int64_t Val = 0) {
    MCValue R;
    assert((!SymB || SymA) && "Invalid relocatable MCValue!");
    R.Cst = Val;
    R.SymA = SymA;
    R.SymB = SymB;
    return R;
  }

  static MCValue get(int64_t Val) {
    MCValue R;
    R.Cst = Val;
    R.SymA = 0;
    R.SymB = 0;
    return R;
  }

};

} // end namespace llvm

#endif
