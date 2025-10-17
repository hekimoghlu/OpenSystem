/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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

//=- llvm/CodeGen/AntiDepBreaker.h - Anti-Dependence Breaking -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AntiDepBreaker class, which implements
// anti-dependence breaking heuristics for post-register-allocation scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ANTIDEPBREAKER_H
#define LLVM_CODEGEN_ANTIDEPBREAKER_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <vector>

namespace llvm {

/// AntiDepBreaker - This class works into conjunction with the
/// post-RA scheduler to rename registers to break register
/// anti-dependencies.
class AntiDepBreaker {
public:
  typedef std::vector<std::pair<MachineInstr *, MachineInstr *> > 
    DbgValueVector;

  virtual ~AntiDepBreaker();

  /// Start - Initialize anti-dep breaking for a new basic block.
  virtual void StartBlock(MachineBasicBlock *BB) =0;

  /// BreakAntiDependencies - Identifiy anti-dependencies within a
  /// basic-block region and break them by renaming registers. Return
  /// the number of anti-dependencies broken.
  ///
  virtual unsigned BreakAntiDependencies(const std::vector<SUnit>& SUnits,
                                         MachineBasicBlock::iterator Begin,
                                         MachineBasicBlock::iterator End,
                                         unsigned InsertPosIndex,
                                         DbgValueVector &DbgValues) = 0;
  
  /// Observe - Update liveness information to account for the current
  /// instruction, which will not be scheduled.
  ///
  virtual void Observe(MachineInstr *MI, unsigned Count,
                       unsigned InsertPosIndex) =0;
  
  /// Finish - Finish anti-dep breaking for a basic block.
  virtual void FinishBlock() =0;

  /// UpdateDbgValue - Update DBG_VALUE if dependency breaker is updating
  /// other machine instruction to use NewReg.
  void UpdateDbgValue(MachineInstr *MI, unsigned OldReg, unsigned NewReg) {
    assert (MI->isDebugValue() && "MI is not DBG_VALUE!");
    if (MI && MI->getOperand(0).isReg() && MI->getOperand(0).getReg() == OldReg)
      MI->getOperand(0).setReg(NewReg);
  }
};

}

#endif
