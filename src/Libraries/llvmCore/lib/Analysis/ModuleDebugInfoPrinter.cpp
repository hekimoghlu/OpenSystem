/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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

//===-- ModuleDebugInfoPrinter.cpp - Prints module debug info metadata ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass decodes the debug info metadata in a module and prints in a
// (sufficiently-prepared-) human-readable form.
//
// For example, run this pass from opt along with the -analyze option, and
// it'll print to standard output.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/DebugInfo.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

namespace {
  class ModuleDebugInfoPrinter : public ModulePass {
    DebugInfoFinder Finder;
  public:
    static char ID; // Pass identification, replacement for typeid
    ModuleDebugInfoPrinter() : ModulePass(ID) {
      initializeModuleDebugInfoPrinterPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
    virtual void print(raw_ostream &O, const Module *M) const;
  };
}

char ModuleDebugInfoPrinter::ID = 0;
INITIALIZE_PASS(ModuleDebugInfoPrinter, "module-debuginfo",
                "Decodes module-level debug info", false, true)

ModulePass *llvm::createModuleDebugInfoPrinterPass() {
  return new ModuleDebugInfoPrinter();
}

bool ModuleDebugInfoPrinter::runOnModule(Module &M) {
  Finder.processModule(M);
  return false;
}

void ModuleDebugInfoPrinter::print(raw_ostream &O, const Module *M) const {
  for (DebugInfoFinder::iterator I = Finder.compile_unit_begin(),
       E = Finder.compile_unit_end(); I != E; ++I) {
    O << "Compile Unit: ";
    DICompileUnit(*I).print(O);
    O << '\n';
  }

  for (DebugInfoFinder::iterator I = Finder.subprogram_begin(),
       E = Finder.subprogram_end(); I != E; ++I) {
    O << "Subprogram: ";
    DISubprogram(*I).print(O);
    O << '\n';
  }

  for (DebugInfoFinder::iterator I = Finder.global_variable_begin(),
       E = Finder.global_variable_end(); I != E; ++I) {
    O << "GlobalVariable: ";
    DIGlobalVariable(*I).print(O);
    O << '\n';
  }

  for (DebugInfoFinder::iterator I = Finder.type_begin(),
       E = Finder.type_end(); I != E; ++I) {
    O << "Type: ";
    DIType(*I).print(O);
    O << '\n';
  }
}
