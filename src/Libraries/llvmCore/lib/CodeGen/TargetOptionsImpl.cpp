/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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

//===-- TargetOptionsImpl.cpp - Options that apply to all targets ----------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the methods in the TargetOptions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

/// DisableFramePointerElim - This returns true if frame pointer elimination
/// optimization should be disabled for the given machine function.
bool TargetOptions::DisableFramePointerElim(const MachineFunction &MF) const {
  // Check to see if we should eliminate non-leaf frame pointers and then
  // check to see if we should eliminate all frame pointers.
  if (NoFramePointerElimNonLeaf && !NoFramePointerElim) {
    const MachineFrameInfo *MFI = MF.getFrameInfo();
    return MFI->hasCalls();
  }

  return NoFramePointerElim;
}

/// LessPreciseFPMAD - This flag return true when -enable-fp-mad option
/// is specified on the command line.  When this flag is off(default), the
/// code generator is not allowed to generate mad (multiply add) if the
/// result is "less precise" than doing those operations individually.
bool TargetOptions::LessPreciseFPMAD() const {
  return UnsafeFPMath || LessPreciseFPMADOption;
}

/// HonorSignDependentRoundingFPMath - Return true if the codegen must assume
/// that the rounding mode of the FPU can change from its default.
bool TargetOptions::HonorSignDependentRoundingFPMath() const {
  return !UnsafeFPMath && HonorSignDependentRoundingFPMathOption;
}

/// getTrapFunctionName - If this returns a non-empty string, this means isel
/// should lower Intrinsic::trap to a call to the specified function name
/// instead of an ISD::TRAP node.
StringRef TargetOptions::getTrapFunctionName() const {
  return TrapFuncName;
}

