/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
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

//===-- XCoreMachineFuctionInfo.h - XCore machine function info -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares XCore-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef XCOREMACHINEFUNCTIONINFO_H
#define XCOREMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include <vector>

namespace llvm {

// Forward declarations
class Function;

/// XCoreFunctionInfo - This class is derived from MachineFunction private
/// XCore target-specific information for each MachineFunction.
class XCoreFunctionInfo : public MachineFunctionInfo {
  virtual void anchor();
  bool UsesLR;
  int LRSpillSlot;
  int FPSpillSlot;
  int VarArgsFrameIndex;
  std::vector<std::pair<MCSymbol*, CalleeSavedInfo> > SpillLabels;

public:
  XCoreFunctionInfo() :
    UsesLR(false),
    LRSpillSlot(0),
    FPSpillSlot(0),
    VarArgsFrameIndex(0) {}
  
  explicit XCoreFunctionInfo(MachineFunction &MF) :
    UsesLR(false),
    LRSpillSlot(0),
    FPSpillSlot(0),
    VarArgsFrameIndex(0) {}
  
  ~XCoreFunctionInfo() {}
  
  void setVarArgsFrameIndex(int off) { VarArgsFrameIndex = off; }
  int getVarArgsFrameIndex() const { return VarArgsFrameIndex; }
  
  void setUsesLR(bool val) { UsesLR = val; }
  bool getUsesLR() const { return UsesLR; }
  
  void setLRSpillSlot(int off) { LRSpillSlot = off; }
  int getLRSpillSlot() const { return LRSpillSlot; }
  
  void setFPSpillSlot(int off) { FPSpillSlot = off; }
  int getFPSpillSlot() const { return FPSpillSlot; }
  
  std::vector<std::pair<MCSymbol*, CalleeSavedInfo> > &getSpillLabels() {
    return SpillLabels;
  }
};
} // End llvm namespace

#endif // XCOREMACHINEFUNCTIONINFO_H
