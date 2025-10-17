/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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

//===---------------- lib/CodeGen/CalcSpillWeights.h ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CODEGEN_CALCSPILLWEIGHTS_H
#define LLVM_CODEGEN_CALCSPILLWEIGHTS_H

#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

  class LiveInterval;
  class LiveIntervals;
  class MachineLoopInfo;

  /// normalizeSpillWeight - The spill weight of a live interval is computed as:
  ///
  ///   (sum(use freq) + sum(def freq)) / (K + size)
  ///
  /// @param UseDefFreq Expected number of executed use and def instructions
  ///                   per function call. Derived from block frequencies.
  /// @param Size       Size of live interval as returnexd by getSize()
  ///
  static inline float normalizeSpillWeight(float UseDefFreq, unsigned Size) {
    // The constant 25 instructions is added to avoid depending too much on
    // accidental SlotIndex gaps for small intervals. The effect is that small
    // intervals have a spill weight that is mostly proportional to the number
    // of uses, while large intervals get a spill weight that is closer to a use
    // density.
    return UseDefFreq / (Size + 25*SlotIndex::InstrDist);
  }

  /// VirtRegAuxInfo - Calculate auxiliary information for a virtual
  /// register such as its spill weight and allocation hint.
  class VirtRegAuxInfo {
    MachineFunction &MF;
    LiveIntervals &LIS;
    const MachineLoopInfo &Loops;
    DenseMap<unsigned, float> Hint;
  public:
    VirtRegAuxInfo(MachineFunction &mf, LiveIntervals &lis,
                   const MachineLoopInfo &loops) :
      MF(mf), LIS(lis), Loops(loops) {}

    /// CalculateWeightAndHint - (re)compute li's spill weight and allocation
    /// hint.
    void CalculateWeightAndHint(LiveInterval &li);
  };

  /// CalculateSpillWeights - Compute spill weights for all virtual register
  /// live intervals.
  class CalculateSpillWeights : public MachineFunctionPass {
  public:
    static char ID;

    CalculateSpillWeights() : MachineFunctionPass(ID) {
      initializeCalculateSpillWeightsPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage &au) const;

    virtual bool runOnMachineFunction(MachineFunction &fn);

  private:
    /// Returns true if the given live interval is zero length.
    bool isZeroLengthInterval(LiveInterval *li) const;
  };

}

#endif // LLVM_CODEGEN_CALCSPILLWEIGHTS_H
