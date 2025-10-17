/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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

//===- AnalysisWrappers.cpp - Wrappers around non-pass analyses -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines pass wrappers around LLVM analyses that don't make sense to
// be passes.  It provides a nice standard pass interface to these classes so
// that they can be printed out by analyze.
//
// These classes are separated out of analyze.cpp so that it is more clear which
// code is the integral part of the analyze tool, and which part of the code is
// just making it so more passes are available.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  /// ExternalFunctionsPassedConstants - This pass prints out call sites to
  /// external functions that are called with constant arguments.  This can be
  /// useful when looking for standard library functions we should constant fold
  /// or handle in alias analyses.
  struct ExternalFunctionsPassedConstants : public ModulePass {
    static char ID; // Pass ID, replacement for typeid
    ExternalFunctionsPassedConstants() : ModulePass(ID) {}
    virtual bool runOnModule(Module &M) {
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
        if (!I->isDeclaration()) continue;
        
        bool PrintedFn = false;
        for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
             UI != E; ++UI) {
          Instruction *User = dyn_cast<Instruction>(*UI);
          if (!User) continue;
          
          CallSite CS(cast<Value>(User));
          if (!CS) continue;
          
          for (CallSite::arg_iterator AI = CS.arg_begin(),
               E = CS.arg_end(); AI != E; ++AI) {
            if (!isa<Constant>(*AI)) continue;

            if (!PrintedFn) {
              errs() << "Function '" << I->getName() << "':\n";
              PrintedFn = true;
            }
            errs() << *User;
            break;
          }
        }
      }

      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char ExternalFunctionsPassedConstants::ID = 0;
static RegisterPass<ExternalFunctionsPassedConstants>
  P1("print-externalfnconstants",
     "Print external fn callsites passed constants");

namespace {
  struct CallGraphPrinter : public ModulePass {
    static char ID; // Pass ID, replacement for typeid
    CallGraphPrinter() : ModulePass(ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequiredTransitive<CallGraph>();
    }
    virtual bool runOnModule(Module &M) {
      getAnalysis<CallGraph>().print(errs(), &M);
      return false;
    }
  };
}

char CallGraphPrinter::ID = 0;
static RegisterPass<CallGraphPrinter>
  P2("print-callgraph", "Print a call graph");
