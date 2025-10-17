/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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

//===-- MachineFunctionAnalysis.h - Owner of MachineFunctions ----*-C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MachineFunctionAnalysis class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINE_FUNCTION_ANALYSIS_H
#define LLVM_CODEGEN_MACHINE_FUNCTION_ANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class MachineFunction;

/// MachineFunctionAnalysis - This class is a Pass that manages a
/// MachineFunction object.
struct MachineFunctionAnalysis : public FunctionPass {
private:
  const TargetMachine &TM;
  MachineFunction *MF;
  unsigned NextFnNum;
public:
  static char ID;
  explicit MachineFunctionAnalysis(const TargetMachine &tm);
  ~MachineFunctionAnalysis();

  MachineFunction &getMF() const { return *MF; }
  
  virtual const char* getPassName() const {
    return "Machine Function Analysis";
  }

private:
  virtual bool doInitialization(Module &M);
  virtual bool runOnFunction(Function &F);
  virtual void releaseMemory();
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
};

} // End llvm namespace

#endif
