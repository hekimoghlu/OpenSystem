/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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

//===-- MachineFunctionPass.h - Pass for MachineFunctions --------*-C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachineFunctionPass class.  MachineFunctionPass's are
// just FunctionPass's, except they operate on machine code as part of a code
// generator.  Because they operate on machine code, not the LLVM
// representation, MachineFunctionPass's are not allowed to modify the LLVM
// representation.  Due to this limitation, the MachineFunctionPass class takes
// care of declaring that no LLVM passes are invalidated.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINE_FUNCTION_PASS_H
#define LLVM_CODEGEN_MACHINE_FUNCTION_PASS_H

#include "llvm/Pass.h"

namespace llvm {

class MachineFunction;

/// MachineFunctionPass - This class adapts the FunctionPass interface to
/// allow convenient creation of passes that operate on the MachineFunction
/// representation. Instead of overriding runOnFunction, subclasses
/// override runOnMachineFunction.
class MachineFunctionPass : public FunctionPass {
protected:
  explicit MachineFunctionPass(char &ID) : FunctionPass(ID) {}

  /// runOnMachineFunction - This method must be overloaded to perform the
  /// desired machine code transformation or analysis.
  ///
  virtual bool runOnMachineFunction(MachineFunction &MF) = 0;

  /// getAnalysisUsage - Subclasses that override getAnalysisUsage
  /// must call this.
  ///
  /// For MachineFunctionPasses, calling AU.preservesCFG() indicates that
  /// the pass does not modify the MachineBasicBlock CFG.
  ///
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

private:
  /// createPrinterPass - Get a machine function printer pass.
  virtual Pass *createPrinterPass(raw_ostream &O,
                                  const std::string &Banner) const;

  virtual bool runOnFunction(Function &F);
};

} // End llvm namespace

#endif
