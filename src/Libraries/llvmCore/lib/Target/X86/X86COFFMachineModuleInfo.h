/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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

//===-- X86coffmachinemoduleinfo.h - X86 COFF MMI Impl ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an MMI implementation for X86 COFF (windows) targets.
//
//===----------------------------------------------------------------------===//

#ifndef X86COFF_MACHINEMODULEINFO_H
#define X86COFF_MACHINEMODULEINFO_H

#include "X86MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {
  class X86MachineFunctionInfo;
  class TargetData;

/// X86COFFMachineModuleInfo - This is a MachineModuleInfoImpl implementation
/// for X86 COFF targets.
class X86COFFMachineModuleInfo : public MachineModuleInfoImpl {
  DenseSet<MCSymbol const *> Externals;
public:
  X86COFFMachineModuleInfo(const MachineModuleInfo &) {}
  virtual ~X86COFFMachineModuleInfo();

  void addExternalFunction(MCSymbol* Symbol) {
    Externals.insert(Symbol);
  }

  typedef DenseSet<MCSymbol const *>::const_iterator externals_iterator;
  externals_iterator externals_begin() const { return Externals.begin(); }
  externals_iterator externals_end() const { return Externals.end(); }
};



} // end namespace llvm

#endif
