/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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

//========-------- BlockFrequencyInfo.h - Block Frequency Analysis -------========//
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

#ifndef LLVM_ANALYSIS_BLOCKFREQUENCYINFO_H
#define LLVM_ANALYSIS_BLOCKFREQUENCYINFO_H

#include "llvm/Pass.h"
#include "llvm/Support/BlockFrequency.h"
#include <climits>

namespace llvm {

class BranchProbabilityInfo;
template<class BlockT, class FunctionT, class BranchProbInfoT>
class BlockFrequencyImpl;

/// BlockFrequencyInfo pass uses BlockFrequencyImpl implementation to estimate
/// IR basic block frequencies.
class BlockFrequencyInfo : public FunctionPass {

  BlockFrequencyImpl<BasicBlock, Function, BranchProbabilityInfo> *BFI;

public:
  static char ID;

  BlockFrequencyInfo();

  ~BlockFrequencyInfo();

  void getAnalysisUsage(AnalysisUsage &AU) const;

  bool runOnFunction(Function &F);
  void print(raw_ostream &O, const Module *M) const;

  /// getblockFreq - Return block frequency. Return 0 if we don't have the
  /// information. Please note that initial frequency is equal to 1024. It means
  /// that we should not rely on the value itself, but only on the comparison to
  /// the other block frequencies. We do this to avoid using of floating points.
  ///
  BlockFrequency getBlockFreq(const BasicBlock *BB) const;
};

}

#endif
