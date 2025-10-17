/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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

//===- MipsJITInfo.h - Mips Implementation of the JIT Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MipsJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSJITINFO_H
#define MIPSJITINFO_H

#include "MipsMachineFunction.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetJITInfo.h"

namespace llvm {
class MipsTargetMachine;

class MipsJITInfo : public TargetJITInfo {

  bool IsPIC;

  public:
    explicit MipsJITInfo() :
      IsPIC(false) {}

    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New);

    // getStubLayout - Returns the size and alignment of the largest call stub
    // on Mips.
    virtual StubLayout getStubLayout();

    /// emitFunctionStub - Use the specified JITCodeEmitter object to emit a
    /// small native function that simply calls the function at the specified
    /// address.
    virtual void *emitFunctionStub(const Function *F, void *Fn,
                                   JITCodeEmitter &JCE);

    /// getLazyResolverFunction - Expose the lazy resolver to the JIT.
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn);

    /// relocate - Before the JIT can run a block of code that has been emitted,
    /// it must rewrite the code to contain the actual addresses of any
    /// referenced global symbols.
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char *GOTBase);

    /// Initialize - Initialize internal stage for the function being JITted.
    void Initialize(const MachineFunction &MF, bool isPIC) {
      IsPIC = isPIC;
    }

};
}

#endif
