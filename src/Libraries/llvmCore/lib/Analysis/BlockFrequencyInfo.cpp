/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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

//=======-------- BlockFrequencyInfo.cpp - Block Frequency Analysis -------=======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Loops should be simplified before this analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"
#include "llvm/Analysis/BlockFrequencyImpl.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"

using namespace llvm;

INITIALIZE_PASS_BEGIN(BlockFrequencyInfo, "block-freq", "Block Frequency Analysis",
                      true, true)
INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfo)
INITIALIZE_PASS_END(BlockFrequencyInfo, "block-freq", "Block Frequency Analysis",
                    true, true)

char BlockFrequencyInfo::ID = 0;


BlockFrequencyInfo::BlockFrequencyInfo() : FunctionPass(ID) {
  initializeBlockFrequencyInfoPass(*PassRegistry::getPassRegistry());
  BFI = new BlockFrequencyImpl<BasicBlock, Function, BranchProbabilityInfo>();
}

BlockFrequencyInfo::~BlockFrequencyInfo() {
  delete BFI;
}

void BlockFrequencyInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<BranchProbabilityInfo>();
  AU.setPreservesAll();
}

bool BlockFrequencyInfo::runOnFunction(Function &F) {
  BranchProbabilityInfo &BPI = getAnalysis<BranchProbabilityInfo>();
  BFI->doFunction(&F, &BPI);
  return false;
}

void BlockFrequencyInfo::print(raw_ostream &O, const Module *) const {
  if (BFI) BFI->print(O);
}

/// getblockFreq - Return block frequency. Return 0 if we don't have the
/// information. Please note that initial frequency is equal to 1024. It means
/// that we should not rely on the value itself, but only on the comparison to
/// the other block frequencies. We do this to avoid using of floating points.
///
BlockFrequency BlockFrequencyInfo::getBlockFreq(const BasicBlock *BB) const {
  return BFI->getBlockFreq(BB);
}
