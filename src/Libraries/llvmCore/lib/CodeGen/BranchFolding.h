/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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

//===-- BranchFolding.h - Fold machine code branch instructions --*- C++ -*===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BRANCHFOLDING_HPP
#define LLVM_CODEGEN_BRANCHFOLDING_HPP

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include <vector>

namespace llvm {
  class MachineFunction;
  class MachineModuleInfo;
  class RegScavenger;
  class TargetInstrInfo;
  class TargetRegisterInfo;

  class BranchFolder {
  public:
    explicit BranchFolder(bool defaultEnableTailMerge, bool CommonHoist);

    bool OptimizeFunction(MachineFunction &MF,
                          const TargetInstrInfo *tii,
                          const TargetRegisterInfo *tri,
                          MachineModuleInfo *mmi);
  private:
    class MergePotentialsElt {
      unsigned Hash;
      MachineBasicBlock *Block;
    public:
      MergePotentialsElt(unsigned h, MachineBasicBlock *b)
        : Hash(h), Block(b) {}

      unsigned getHash() const { return Hash; }
      MachineBasicBlock *getBlock() const { return Block; }

      void setBlock(MachineBasicBlock *MBB) {
        Block = MBB;
      }

      bool operator<(const MergePotentialsElt &) const;
    };
    typedef std::vector<MergePotentialsElt>::iterator MPIterator;
    std::vector<MergePotentialsElt> MergePotentials;
    SmallPtrSet<const MachineBasicBlock*, 2> TriedMerging;

    class SameTailElt {
      MPIterator MPIter;
      MachineBasicBlock::iterator TailStartPos;
    public:
      SameTailElt(MPIterator mp, MachineBasicBlock::iterator tsp)
        : MPIter(mp), TailStartPos(tsp) {}

      MPIterator getMPIter() const {
        return MPIter;
      }
      MergePotentialsElt &getMergePotentialsElt() const {
        return *getMPIter();
      }
      MachineBasicBlock::iterator getTailStartPos() const {
        return TailStartPos;
      }
      unsigned getHash() const {
        return getMergePotentialsElt().getHash();
      }
      MachineBasicBlock *getBlock() const {
        return getMergePotentialsElt().getBlock();
      }
      bool tailIsWholeBlock() const {
        return TailStartPos == getBlock()->begin();
      }

      void setBlock(MachineBasicBlock *MBB) {
        getMergePotentialsElt().setBlock(MBB);
      }
      void setTailStartPos(MachineBasicBlock::iterator Pos) {
        TailStartPos = Pos;
      }
    };
    std::vector<SameTailElt> SameTails;

    bool EnableTailMerge;
    bool EnableHoistCommonCode;
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    MachineModuleInfo *MMI;
    RegScavenger *RS;

    bool TailMergeBlocks(MachineFunction &MF);
    bool TryTailMergeBlocks(MachineBasicBlock* SuccBB,
                       MachineBasicBlock* PredBB);
    void MaintainLiveIns(MachineBasicBlock *CurMBB,
                         MachineBasicBlock *NewMBB);
    void ReplaceTailWithBranchTo(MachineBasicBlock::iterator OldInst,
                                 MachineBasicBlock *NewDest);
    MachineBasicBlock *SplitMBBAt(MachineBasicBlock &CurMBB,
                                  MachineBasicBlock::iterator BBI1);
    unsigned ComputeSameTails(unsigned CurHash, unsigned minCommonTailLength,
                              MachineBasicBlock *SuccBB,
                              MachineBasicBlock *PredBB);
    void RemoveBlocksWithHash(unsigned CurHash, MachineBasicBlock* SuccBB,
                                                MachineBasicBlock* PredBB);
    bool CreateCommonTailOnlyBlock(MachineBasicBlock *&PredBB,
                                   unsigned maxCommonTailLength,
                                   unsigned &commonTailIndex);

    bool OptimizeBranches(MachineFunction &MF);
    bool OptimizeBlock(MachineBasicBlock *MBB);
    void RemoveDeadBlock(MachineBasicBlock *MBB);
    bool OptimizeImpDefsBlock(MachineBasicBlock *MBB);

    bool HoistCommonCode(MachineFunction &MF);
    bool HoistCommonCodeInSuccs(MachineBasicBlock *MBB);
  };
}

#endif /* LLVM_CODEGEN_BRANCHFOLDING_HPP */
