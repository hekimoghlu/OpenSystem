/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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

//===-- PPCJITInfo.h - PowerPC impl. of the JIT interface -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetJITInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC_JITINFO_H
#define POWERPC_JITINFO_H

#include "llvm/Target/TargetJITInfo.h"
#include "llvm/CodeGen/JITCodeEmitter.h"

namespace llvm {
  class PPCTargetMachine;

  class PPCJITInfo : public TargetJITInfo {
  protected:
    PPCTargetMachine &TM;
    bool is64Bit;
  public:
    PPCJITInfo(PPCTargetMachine &tm, bool tmIs64Bit) : TM(tm) {
      useGOT = 0;
      is64Bit = tmIs64Bit;
    }

    virtual StubLayout getStubLayout();
    virtual void *emitFunctionStub(const Function* F, void *Fn,
                                   JITCodeEmitter &JCE);
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn);
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase);
    
    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New);
  };
}

#endif
