/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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

//===-- SPUMachineFunctionInfo.h - Private data used for CellSPU --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the IBM Cell SPU specific subclass of MachineFunctionInfo.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_MACHINE_FUNCTION_INFO_H
#define SPU_MACHINE_FUNCTION_INFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// SPUFunctionInfo - Cell SPU target-specific information for each
/// MachineFunction
class SPUFunctionInfo : public MachineFunctionInfo {
  virtual void anchor();

  /// UsesLR - Indicates whether LR is used in the current function.
  ///
  bool UsesLR;

  // VarArgsFrameIndex - FrameIndex for start of varargs area.
  int VarArgsFrameIndex;

public:
  SPUFunctionInfo(MachineFunction& MF) 
  : UsesLR(false),
    VarArgsFrameIndex(0)
  {}

  void setUsesLR(bool U) { UsesLR = U; }
  bool usesLR()          { return UsesLR; }

  int getVarArgsFrameIndex() const { return VarArgsFrameIndex; }
  void setVarArgsFrameIndex(int Index) { VarArgsFrameIndex = Index; }
};

} // end of namespace llvm


#endif

