/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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

//===-- MachineFunctionAnalysis.cpp ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the MachineFunctionAnalysis members.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
using namespace llvm;

char MachineFunctionAnalysis::ID = 0;

MachineFunctionAnalysis::MachineFunctionAnalysis(const TargetMachine &tm) :
  FunctionPass(ID), TM(tm), MF(0) {
  initializeMachineModuleInfoPass(*PassRegistry::getPassRegistry());
}

MachineFunctionAnalysis::~MachineFunctionAnalysis() {
  releaseMemory();
  assert(!MF && "MachineFunctionAnalysis left initialized!");
}

void MachineFunctionAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineModuleInfo>();
}

bool MachineFunctionAnalysis::doInitialization(Module &M) {
  MachineModuleInfo *MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  assert(MMI && "MMI not around yet??");
  MMI->setModule(&M);
  NextFnNum = 0;
  return false;
}


bool MachineFunctionAnalysis::runOnFunction(Function &F) {
  assert(!MF && "MachineFunctionAnalysis already initialized!");
  MF = new MachineFunction(&F, TM, NextFnNum++,
                           getAnalysis<MachineModuleInfo>(),
                           getAnalysisIfAvailable<GCModuleInfo>());
  return false;
}

void MachineFunctionAnalysis::releaseMemory() {
  delete MF;
  MF = 0;
}
