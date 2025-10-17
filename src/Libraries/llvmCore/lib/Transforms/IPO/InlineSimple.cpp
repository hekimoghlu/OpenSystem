/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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

//===- InlineSimple.cpp - Code to perform simple function inlining --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements bottom-up inlining of functions into callees.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "inline"
#include "llvm/CallingConv.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

namespace {

  class SimpleInliner : public Inliner {
    InlineCostAnalyzer CA;
  public:
    SimpleInliner() : Inliner(ID) {
      initializeSimpleInlinerPass(*PassRegistry::getPassRegistry());
    }
    SimpleInliner(int Threshold) : Inliner(ID, Threshold,
                                           /*InsertLifetime*/true) {
      initializeSimpleInlinerPass(*PassRegistry::getPassRegistry());
    }
    static char ID; // Pass identification, replacement for typeid
    InlineCost getInlineCost(CallSite CS) {
      return CA.getInlineCost(CS, getInlineThreshold(CS));
    }
    virtual bool doInitialization(CallGraph &CG);
  };
}

char SimpleInliner::ID = 0;
INITIALIZE_PASS_BEGIN(SimpleInliner, "inline",
                "Function Integration/Inlining", false, false)
INITIALIZE_AG_DEPENDENCY(CallGraph)
INITIALIZE_PASS_END(SimpleInliner, "inline",
                "Function Integration/Inlining", false, false)

Pass *llvm::createFunctionInliningPass() { return new SimpleInliner(); }

Pass *llvm::createFunctionInliningPass(int Threshold) {
  return new SimpleInliner(Threshold);
}

// doInitialization - Initializes the vector of functions that have been
// annotated with the noinline attribute.
bool SimpleInliner::doInitialization(CallGraph &CG) {
  CA.setTargetData(getAnalysisIfAvailable<TargetData>());
  return false;
}

