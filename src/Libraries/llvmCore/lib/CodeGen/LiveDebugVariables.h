/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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

//===- LiveDebugVariables.h - Tracking debug info variables ----*- c++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface to the LiveDebugVariables analysis.
//
// The analysis removes DBG_VALUE instructions for virtual registers and tracks
// live user variables in a data structure that can be updated during register
// allocation.
//
// After register allocation new DBG_VALUE instructions are emitted to reflect
// the new locations of user variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEDEBUGVARIABLES_H
#define LLVM_CODEGEN_LIVEDEBUGVARIABLES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

class LiveInterval;
class VirtRegMap;

class LiveDebugVariables : public MachineFunctionPass {
  void *pImpl;
public:
  static char ID; // Pass identification, replacement for typeid

  LiveDebugVariables();
  ~LiveDebugVariables();

  /// renameRegister - Move any user variables in OldReg to NewReg:SubIdx.
  /// @param OldReg Old virtual register that is going away.
  /// @param NewReg New register holding the user variables.
  /// @param SubIdx If NewReg is a virtual register, SubIdx may indicate a sub-
  ///               register.
  void renameRegister(unsigned OldReg, unsigned NewReg, unsigned SubIdx);

  /// splitRegister - Move any user variables in OldReg to the live ranges in
  /// NewRegs where they are live. Mark the values as unavailable where no new
  /// register is live.
  void splitRegister(unsigned OldReg, ArrayRef<LiveInterval*> NewRegs);

  /// emitDebugValues - Emit new DBG_VALUE instructions reflecting the changes
  /// that happened during register allocation.
  /// @param VRM Rename virtual registers according to map.
  void emitDebugValues(VirtRegMap *VRM);

  /// dump - Print data structures to dbgs().
  void dump();

private:

  virtual bool runOnMachineFunction(MachineFunction &);
  virtual void releaseMemory();
  virtual void getAnalysisUsage(AnalysisUsage &) const;

};

} // namespace llvm

#endif // LLVM_CODEGEN_LIVEDEBUGVARIABLES_H
