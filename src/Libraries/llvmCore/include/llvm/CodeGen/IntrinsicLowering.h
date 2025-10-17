/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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

//===-- IntrinsicLowering.h - Intrinsic Function Lowering -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the IntrinsicLowering interface.  This interface allows
// addition of domain-specific or front-end specific intrinsics to LLVM without
// having to modify all of the C backend or interpreter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_INTRINSICLOWERING_H
#define LLVM_CODEGEN_INTRINSICLOWERING_H

#include "llvm/Intrinsics.h"

namespace llvm {
  class CallInst;
  class Module;
  class TargetData;

  class IntrinsicLowering {
    const TargetData& TD;

    
    bool Warned;
  public:
    explicit IntrinsicLowering(const TargetData &td) :
      TD(td), Warned(false) {}

    /// AddPrototypes - This method, if called, causes all of the prototypes
    /// that might be needed by an intrinsic lowering implementation to be
    /// inserted into the module specified.
    void AddPrototypes(Module &M);

    /// LowerIntrinsicCall - This method replaces a call with the LLVM function
    /// which should be used to implement the specified intrinsic function call.
    /// If an intrinsic function must be implemented by the code generator 
    /// (such as va_start), this function should print a message and abort.
    ///
    /// Otherwise, if an intrinsic function call can be lowered, the code to
    /// implement it (often a call to a non-intrinsic function) is inserted
    /// _after_ the call instruction and the call is deleted.  The caller must
    /// be capable of handling this kind of change.
    ///
    void LowerIntrinsicCall(CallInst *CI);

    /// LowerToByteSwap - Replace a call instruction into a call to bswap
    /// intrinsic. Return false if it has determined the call is not a
    /// simple integer bswap.
    static bool LowerToByteSwap(CallInst *CI);
  };
}

#endif
