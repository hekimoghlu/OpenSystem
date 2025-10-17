/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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

//===-- llvm/MC/MCCodeGenInfo.h - Target CodeGen Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tracks information about the target which can affect codegen,
// asm parsing, and asm printing. For example, relocation model.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCCODEGENINFO_H
#define LLVM_MC_MCCODEGENINFO_H

#include "llvm/Support/CodeGen.h"

namespace llvm {

  class MCCodeGenInfo {
    /// RelocationModel - Relocation model: static, pic, etc.
    ///
    Reloc::Model RelocationModel;

    /// CMModel - Code model.
    ///
    CodeModel::Model CMModel;

    /// OptLevel - Optimization level.
    ///
    CodeGenOpt::Level OptLevel;

  public:
    void InitMCCodeGenInfo(Reloc::Model RM = Reloc::Default,
                           CodeModel::Model CM = CodeModel::Default,
                           CodeGenOpt::Level OL = CodeGenOpt::Default);

    Reloc::Model getRelocationModel() const { return RelocationModel; }

    CodeModel::Model getCodeModel() const { return CMModel; }

    CodeGenOpt::Level getOptLevel() const { return OptLevel; }
  };
} // namespace llvm

#endif
