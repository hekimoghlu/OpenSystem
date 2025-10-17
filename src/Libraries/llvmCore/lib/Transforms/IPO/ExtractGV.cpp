/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

//===-- ExtractGV.cpp - Global Value extraction pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass extracts global values
//
//===----------------------------------------------------------------------===//

#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Constants.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>
using namespace llvm;

namespace {
  /// @brief A pass to extract specific functions and their dependencies.
  class GVExtractorPass : public ModulePass {
    SetVector<GlobalValue *> Named;
    bool deleteStuff;
  public:
    static char ID; // Pass identification, replacement for typeid

    /// FunctionExtractorPass - If deleteFn is true, this pass deletes as the
    /// specified function. Otherwise, it deletes as much of the module as
    /// possible, except for the function specified.
    ///
    explicit GVExtractorPass(std::vector<GlobalValue*>& GVs, bool deleteS = true)
      : ModulePass(ID), Named(GVs.begin(), GVs.end()), deleteStuff(deleteS) {}

    bool runOnModule(Module &M) {
      // Visit the global inline asm.
      if (!deleteStuff)
        M.setModuleInlineAsm("");

      // For simplicity, just give all GlobalValues ExternalLinkage. A trickier
      // implementation could figure out which GlobalValues are actually
      // referenced by the Named set, and which GlobalValues in the rest of
      // the module are referenced by the NamedSet, and get away with leaving
      // more internal and private things internal and private. But for now,
      // be conservative and simple.

      // Visit the GlobalVariables.
      for (Module::global_iterator I = M.global_begin(), E = M.global_end();
           I != E; ++I) {
        if (deleteStuff == (bool)Named.count(I) && !I->isDeclaration()) {
          I->setInitializer(0);
        } else {
          if (I->hasAvailableExternallyLinkage())
            continue;
          if (I->getName() == "llvm.global_ctors")
            continue;
        }

        if (I->hasLocalLinkage())
          I->setVisibility(GlobalValue::HiddenVisibility);
        I->setLinkage(GlobalValue::ExternalLinkage);
      }

      // Visit the Functions.
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
        if (deleteStuff == (bool)Named.count(I) && !I->isDeclaration()) {
          I->deleteBody();
        } else {
          if (I->hasAvailableExternallyLinkage())
            continue;
        }

        if (I->hasLocalLinkage())
          I->setVisibility(GlobalValue::HiddenVisibility);
        I->setLinkage(GlobalValue::ExternalLinkage);
      }

      return true;
    }
  };

  char GVExtractorPass::ID = 0;
}

ModulePass *llvm::createGVExtractionPass(std::vector<GlobalValue*>& GVs, 
                                         bool deleteFn) {
  return new GVExtractorPass(GVs, deleteFn);
}
