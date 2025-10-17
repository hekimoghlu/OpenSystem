/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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

//===- LibCallAliasAnalysis.h - Implement AliasAnalysis for libcalls ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LibCallAliasAnalysis class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LIBCALL_AA_H
#define LLVM_ANALYSIS_LIBCALL_AA_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"

namespace llvm {
  class LibCallInfo;
  struct LibCallFunctionInfo;
  
  /// LibCallAliasAnalysis - Alias analysis driven from LibCallInfo.
  struct LibCallAliasAnalysis : public FunctionPass, public AliasAnalysis {
    static char ID; // Class identification
    
    LibCallInfo *LCI;
    
    explicit LibCallAliasAnalysis(LibCallInfo *LC = 0)
        : FunctionPass(ID), LCI(LC) {
      initializeLibCallAliasAnalysisPass(*PassRegistry::getPassRegistry());
    }
    explicit LibCallAliasAnalysis(char &ID, LibCallInfo *LC)
        : FunctionPass(ID), LCI(LC) {
      initializeLibCallAliasAnalysisPass(*PassRegistry::getPassRegistry());
    }
    ~LibCallAliasAnalysis();
    
    ModRefResult getModRefInfo(ImmutableCallSite CS,
                               const Location &Loc);
    
    ModRefResult getModRefInfo(ImmutableCallSite CS1,
                               ImmutableCallSite CS2) {
      // TODO: Could compare two direct calls against each other if we cared to.
      return AliasAnalysis::getModRefInfo(CS1, CS2);
    }
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    
    virtual bool runOnFunction(Function &F) {
      InitializeAliasAnalysis(this);                 // set up super class
      return false;
    }
    
    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    virtual void *getAdjustedAnalysisPointer(const void *PI) {
      if (PI == &AliasAnalysis::ID)
        return (AliasAnalysis*)this;
      return this;
    }
    
  private:
    ModRefResult AnalyzeLibCallDetails(const LibCallFunctionInfo *FI,
                                       ImmutableCallSite CS,
                                       const Location &Loc);
  };
}  // End of llvm namespace

#endif
