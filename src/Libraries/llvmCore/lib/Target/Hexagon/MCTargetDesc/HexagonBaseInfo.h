/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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

//===-- HexagonBaseInfo.h - Top level definitions for Hexagon --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the Hexagon target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBASEINFO_H
#define HEXAGONBASEINFO_H

namespace llvm {

/// HexagonII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace HexagonII {
  // *** The code below must match HexagonInstrFormat*.td *** //

  // Insn types.
  // *** Must match HexagonInstrFormat*.td ***
  enum Type {
    TypePSEUDO = 0,
    TypeALU32  = 1,
    TypeCR     = 2,
    TypeJR     = 3,
    TypeJ      = 4,
    TypeLD     = 5,
    TypeST     = 6,
    TypeSYSTEM = 7,
    TypeXTYPE  = 8,
    TypeMEMOP  = 9,
    TypeNV     = 10,
    TypePREFIX = 30, // Such as extenders.
    TypeMARKER = 31  // Such as end of a HW loop.
  };



  // MCInstrDesc TSFlags
  // *** Must match HexagonInstrFormat*.td ***
  enum {
    // This 5-bit field describes the insn type.
    TypePos  = 0,
    TypeMask = 0x1f,

    // Solo instructions.
    SoloPos  = 5,
    SoloMask = 0x1,

    // Predicated instructions.
    PredicatedPos  = 6,
    PredicatedMask = 0x1
  };

  // *** The code above must match HexagonInstrFormat*.td *** //

} // End namespace HexagonII.

} // End namespace llvm.

#endif
