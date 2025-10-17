/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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

//===-- MipsDirectObjLower.h - Mips LLVM direct object lowering *- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSDIRECTOBJLOWER_H
#define MIPSDIRECTOBJLOWER_H
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
  class MCInst;
  class MCStreamer;

  namespace Mips {
  /// MipsDirectObjLower - This name space is used to lower MCInstr in cases
  //                       where the assembler usually finishes the lowering
  //                       such as large shifts.
    void LowerLargeShift(MCInst &Inst);
    void LowerDextDins(MCInst &Inst);
  }
}

#endif
