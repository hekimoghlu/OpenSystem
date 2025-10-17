/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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

//===--- Bitcode/Writer/BitcodeWriterPass.cpp - Bitcode Writer ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// BitcodeWriterPass implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Pass.h"
using namespace llvm;

namespace {
  class WriteBitcodePass : public ModulePass {
    raw_ostream &OS; // raw_ostream to print on
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit WriteBitcodePass(raw_ostream &o)
      : ModulePass(ID), OS(o) {}
    
    const char *getPassName() const { return "Bitcode Writer"; }
    
    bool runOnModule(Module &M) {
      WriteBitcodeToFile(&M, OS);
      return false;
    }
  };
}

char WriteBitcodePass::ID = 0;

/// createBitcodeWriterPass - Create and return a pass that writes the module
/// to the specified ostream.
ModulePass *llvm::createBitcodeWriterPass(raw_ostream &Str) {
  return new WriteBitcodePass(Str);
}
